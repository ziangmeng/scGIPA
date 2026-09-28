"""The public-name migration must preserve existing numerical results."""
import unittest
import torch
from scgipa import ScGIPA, scgipa_loss, intervention_effects
from grs_vae import EGRDM
from grs_vae.model import egrdm_loss

class Compatibility(unittest.TestCase):
    def test_legacy_names_and_state_dict_remain_compatible(self):
        self.assertIs(EGRDM, ScGIPA)
        self.assertIs(egrdm_loss, scgipa_loss)
        torch.manual_seed(5)
        prior=torch.tensor([[1., -.5, 0., .8]])
        old=EGRDM(4,1,prior,hidden_dim=16,latent_dim=4).eval()
        new=ScGIPA(4,1,prior,hidden_dim=16,latent_dim=4).eval()
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.rand(2,4);cell=torch.zeros(2,dtype=torch.long);ref=torch.rand(4)
        for a,b in zip(intervention_effects(old,x,cell,ref,[0],0,1),
                       intervention_effects(new,x,cell,ref,[0],0,1)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
