"""Scientific invariants and data-contract checks for the reference workflow."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scgipa import ScGIPA, intervention_effects
from scgipa.reference_data import make_reference_dataset, load_reference_dataset, pseudobulk

class Interventions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(11)
        self.prior=torch.tensor([[1.,-.5,0.,.8],[.7,0.,-.3,1.]])
        self.model=ScGIPA(4,2,self.prior,hidden_dim=16,latent_dim=4).eval()
        self.x=torch.rand(3,4);self.cell=torch.tensor([0,1,0]);self.reference=torch.rand(4)

    def test_interventions_match_explicit_branch_calculation(self):
        before=self.x.clone();prior_before=self.model.prior.clone()
        I,M=intervention_effects(self.model,self.x,self.cell,self.reference,[0,2],1,2)
        with torch.no_grad():
            mu,_=self.model.encode(self.x)
            base=self.model(self.x,self.cell,stochastic=False)[1]
            reset=self.x.clone();reset[:,[0,2]]=self.reference[[0,2]]
            reset_logits=self.model(reset,self.cell,stochastic=False)[1]
            masked=self.x*self.model.prior[self.cell];masked[:,[0,2]]=0
            mask_logits=self.model.classifier(mu+self.model.cell_emb(self.cell)+self.model.route_encoder(masked))
            margin=lambda z:z[:,2]-z[:,1]
        torch.testing.assert_close(I,margin(base)-margin(reset_logits))
        torch.testing.assert_close(M,margin(base)-margin(mask_logits))
        torch.testing.assert_close(self.x,before);torch.testing.assert_close(self.model.prior,prior_before)

    def test_empty_intervention_and_identical_reference(self):
        I,M=intervention_effects(self.model,self.x,self.cell,self.reference,[],0,1)
        torch.testing.assert_close(I,torch.zeros(3));torch.testing.assert_close(M,torch.zeros(3))
        I,_=intervention_effects(self.model,self.x,self.cell,self.x,[0,1,2,3],0,1)
        torch.testing.assert_close(I,torch.zeros(3))

    def test_zero_route_support_has_zero_M(self):
        cell=torch.zeros(3,dtype=torch.long)
        _,M=intervention_effects(self.model,self.x,cell,self.reference,[2],0,1)
        torch.testing.assert_close(M,torch.zeros(3))

    def test_logits_need_no_disease_label_and_are_deterministic(self):
        one=self.model(self.x,self.cell)[1];two=self.model(self.x,self.cell)[1]
        self.assertEqual(one.shape,(3,4));torch.testing.assert_close(one,two)

class DataContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        make_reference_dataset(self.path)
    def tearDown(self):self.temp.cleanup()

    def test_donor_isolation_and_pseudobulk(self):
        counts,meta,prior,genes,cells=load_reference_dataset(self.path)
        self.assertEqual(counts.shape,(864,64));self.assertEqual(meta.donor_id.nunique(),24)
        self.assertTrue((meta.groupby('donor_id').split.nunique()==1).all())
        pb,pm=pseudobulk(counts,meta)
        self.assertEqual(pb.shape,(72,64));self.assertEqual(prior.shape,(3,64))
        row=pm.iloc[0];idx=(meta.donor_id==row.donor_id)&(meta.cell_type==row.cell_type)
        total=counts[idx].sum(0);expected=np.log1p(total/total.sum()*1e4)
        np.testing.assert_allclose(pb[0],expected,rtol=1e-6)

    def test_rejects_donor_leakage(self):
        m=pd.read_csv(self.path/'metadata.csv');m.loc[0,'split']='test';m.to_csv(self.path/'metadata.csv',index=False)
        with self.assertRaisesRegex(ValueError,'donor'):load_reference_dataset(self.path)

    def test_cell_order_is_aligned_by_identifier(self):
        before=load_reference_dataset(self.path)
        m=pd.read_csv(self.path/'metadata.csv').iloc[::-1];m.to_csv(self.path/'metadata.csv',index=False)
        after=load_reference_dataset(self.path)
        pd.testing.assert_frame_equal(before[1],after[1])

if __name__=='__main__':unittest.main()
