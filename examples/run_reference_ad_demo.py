"""Run the four-state EGRDM reference workflow with donor-disjoint inputs."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grs_vae.model import EGRDM, egrdm_loss, intervention_effects
from grs_vae.reference_data import STATES, CONTRASTS, normalise_counts, make_reference_dataset, load_reference_dataset, pseudobulk

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/reference_ad")
    parser.add_argument("--outdir", type=Path, default=ROOT/"outputs/reference_ad_demo")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--regenerate", action="store_true", help="Generate synthetic inputs inside the output directory.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.regenerate:
        args.data_dir = args.outdir/"synthetic_inputs"
        make_reference_dataset(args.data_dir, args.seed)
    counts, meta, prior, genes, cells = load_reference_dataset(args.data_dir)
    cell_ids = {c:i for i,c in enumerate(cells)}
    x = torch.tensor(normalise_counts(counts))
    cell = torch.tensor(meta.cell_type.map(cell_ids).to_numpy(), dtype=torch.long)
    state = torch.tensor(meta.state.map({s:i for i,s in enumerate(STATES)}).to_numpy(), dtype=torch.long)
    train = np.flatnonzero(meta.split.eq("train"))
    validation = np.flatnonzero(meta.split.eq("validation"))
    pb, pm = pseudobulk(counts, meta)
    targets = {}
    for c in cells:
        for a,b in CONTRASTS:
            aa = pb[pm.cell_type.eq(c)&pm.split.eq("train")&pm.state.eq(STATES[a])]
            bb = pb[pm.cell_type.eq(c)&pm.split.eq("train")&pm.state.eq(STATES[b])]
            if len(aa)>=2 and len(bb)>=2:
                targets[cell_ids[c],f"{STATES[a]}_to_{STATES[b]}"] = bb.mean(0)-aa.mean(0)
    model = EGRDM(len(genes),len(cells),prior,hidden_dim=args.hidden_dim,latent_dim=args.latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    best, saved, history = float("inf"), None, []
    selected_epoch = None
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for indices in np.array_split(np.random.permutation(train),max(1,int(np.ceil(len(train)/128)))):
            loss=egrdm_loss(model,model(x[indices],cell[indices]),x[indices],state[indices],targets)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            rec,logits,stage,_,_=model(x[validation],cell[validation],stochastic=False)
            from torch.nn import functional as F
            mask=state[validation]>=1
            vloss=F.mse_loss(rec,x[validation])+F.cross_entropy(logits,state[validation])+.2*F.mse_loss(stage[mask],state[validation][mask].float()-1)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=float(vloss)))
        if float(vloss)<best:
            best=float(vloss);saved=copy.deepcopy(model.state_dict());selected_epoch=epoch
        if epoch in {1,args.epochs}:
            print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} validation_loss={best:.4f}")
    model.load_state_dict(saved);model.eval()
    # Evaluation happens after checkpoint selection, using test donors only.
    test_mask=pm.split.eq("test").to_numpy()
    test_meta=pm.loc[test_mask].reset_index(drop=True)
    test_pb=torch.tensor(pb[test_mask])
    test_cell=torch.tensor(test_meta.cell_type.map(cell_ids).to_numpy(),dtype=torch.long)
    with torch.no_grad():
        probability=model(test_pb,test_cell,stochastic=False)[1].softmax(1).numpy()
    predictions=test_meta[["donor_id","cell_type","state"]].copy()
    for i,s in enumerate(STATES):predictions[f"prob_{s}"]=probability[:,i]
    predictions.to_csv(args.outdir/"test_donor_cell_predictions.csv",index=False)
    donor=predictions.groupby(["donor_id","state"],as_index=False)[[f"prob_{s}" for s in STATES]].mean()
    donor["prediction"]=[STATES[i] for i in donor[[f"prob_{s}" for s in STATES]].to_numpy().argmax(1)]
    donor.to_csv(args.outdir/"test_donor_predictions.csv",index=False)
    effects=[]
    for c in cells:
        for a,b in CONTRASTS:
            reference=pb[pm.cell_type.eq(c)&pm.state.eq(STATES[a])&pm.split.eq("train")]
            ti=np.flatnonzero(pm.cell_type.eq(c)&pm.state.eq(STATES[b])&pm.split.eq("test"))
            if len(reference)<2 or not len(ti):continue
            xx=torch.tensor(pb[ti]);cc=torch.full((len(ti),),cell_ids[c],dtype=torch.long)
            ref=torch.tensor(reference.mean(0))
            selections=[(gene,[g]) for g,gene in enumerate(genes)]
            route_genes=np.flatnonzero(prior[cell_ids[c]]).tolist()
            if route_genes:selections.append(("JOINT_SUPPORTED_ROUTES",route_genes))
            for name,selected in selections:
                I,M=intervention_effects(model,xx,cc,ref,selected,a,b)
                for k,idx in enumerate(ti):
                    effects.append(dict(donor_id=pm.iloc[idx].donor_id,cell_type=c,contrast=f"{STATES[a]}_to_{STATES[b]}",
                                        gene_or_set=name,n_genes=len(selected),I=float(I[k]),M=float(M[k]),n_reference_donors=len(reference)))
    details=pd.DataFrame(effects)
    details.to_csv(args.outdir/"interventions_by_donor.csv",index=False)
    details.groupby(["cell_type","contrast","gene_or_set","n_genes"],as_index=False).agg(
        I=("I","mean"),M=("M","mean"),n_target_donors=("donor_id","nunique")).to_csv(args.outdir/"interventions_summary.csv",index=False)
    pd.DataFrame(history).to_csv(args.outdir/"training_history.csv",index=False)
    meta[["donor_id","state","split"]].drop_duplicates().to_csv(args.outdir/"donor_splits.csv",index=False)
    torch.save(dict(state_dict=saved,genes=genes,cell_types=cells,states=STATES,
                    hidden_dim=args.hidden_dim,latent_dim=args.latent_dim),args.outdir/"model.pt")
    files=["expression_counts.csv","metadata.csv","route_prior_matrix.csv"]
    run=dict(seed=args.seed,epochs=args.epochs,selected_epoch=selected_epoch,n_cells=len(meta),n_genes=len(genes),
             n_donors=meta.donor_id.nunique(),test_donors=len(donor),
             test_accuracy=float(donor.prediction.eq(donor.state).mean()),
             hidden_dim=args.hidden_dim,latent_dim=args.latent_dim,learning_rate=args.learning_rate,
             input_sha256={n:hashlib.sha256((args.data_dir/n).read_bytes()).hexdigest() for n in files},
             software=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),
             evaluation="Single predefined donor split; test donors not used for checkpoint selection.")
    (args.outdir/"run_manifest.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    print(f"Completed: {args.outdir}; test donor accuracy={run['test_accuracy']:.3f}")

if __name__=="__main__":
    main()
