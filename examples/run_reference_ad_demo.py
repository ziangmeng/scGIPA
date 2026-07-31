"""Train the public GRS-VAE reference workflow and write compact outputs."""

from pathlib import Path
import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grs_vae.model import GRSVAE, grs_vae_loss
from grs_vae.reference_data import CELL_TYPES, STATES, load_reference_dataset, make_reference_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "reference_ad_demo")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the bundled synthetic inputs from the fixed seed.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.regenerate:
        counts, metadata, route_prior, genes, prior_table = make_reference_dataset(seed=args.seed)
    else:
        counts, metadata, route_prior, genes, prior_table = load_reference_dataset(ROOT / "data" / "reference_ad")
    metadata.to_csv(args.outdir / "reference_cell_metadata.csv", index=False)
    prior_table.to_csv(args.outdir / "reference_route_prior.csv", index=False)

    expression = torch.tensor(np.log1p(counts), dtype=torch.float32)
    prior = torch.tensor(route_prior, dtype=torch.float32)
    cell = torch.tensor(metadata["cell_type"].map({x: i for i, x in enumerate(CELL_TYPES)}).to_numpy())
    state = torch.tensor(metadata["state"].map({x: i for i, x in enumerate(STATES)}).to_numpy())
    stage = torch.tensor(metadata["stage"].to_numpy(), dtype=torch.float32)

    model = GRSVAE(n_genes=len(genes), n_cell_types=len(CELL_TYPES), n_states=len(STATES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        output = model(expression, cell, state, prior)
        loss, parts = grs_vae_loss(output, expression, state, stage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in {0, args.epochs - 1}:
            print(f"epoch={epoch + 1:03d} loss={loss.item():.4f} recon={parts['reconstruction'].item():.4f}")

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(expression)
        full = model.decode(mu, cell, state, prior)
        masked = model.decode(mu, cell, state, torch.zeros_like(prior))
        effect = (full - masked).numpy()

    route_rows = []
    for cell_id, cell_type in enumerate(CELL_TYPES):
        in_cell = metadata["cell_type"].eq(cell_type).to_numpy()
        control = in_cell & metadata["state"].eq("Control").to_numpy()
        ad = in_cell & metadata["state"].eq("AD").to_numpy()
        state_direction = expression[ad].mean(0).numpy() - expression[control].mean(0).numpy()
        for gene_idx in np.flatnonzero(route_prior[in_cell].max(axis=0) > 0):
            score = float(np.mean(effect[in_cell, gene_idx]) * state_direction[gene_idx])
            route_rows.append({
                "cell_type": cell_type,
                "gene": genes[gene_idx],
                "route_weight": float(route_prior[in_cell, gene_idx].max()),
                "route_masking_score": score,
                "mean_log_expression_control": float(expression[control, gene_idx].mean()),
                "mean_log_expression_ad": float(expression[ad, gene_idx].mean()),
            })

    scores = pd.DataFrame(route_rows).sort_values("route_masking_score", ascending=False)
    scores.to_csv(args.outdir / "reference_route_scores.csv", index=False)
    summary = scores.groupby("cell_type", as_index=False).agg(
        mean_route_score=("route_masking_score", "mean"),
        max_route_score=("route_masking_score", "max"),
        n_routes=("gene", "size"),
    )
    summary.to_csv(args.outdir / "reference_celltype_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = scores.sort_values("route_masking_score")
    colors = plot["cell_type"].map({"Microglia": "#4C78A8", "Astrocytes": "#59A14F", "Excitatory_neurons": "#E15759"})
    ax.barh(plot["cell_type"] + " | " + plot["gene"], plot["route_masking_score"], color=colors)
    ax.axvline(0, color="#404040", lw=0.8)
    ax.set_xlabel("Route-masking score (synthetic reference data)")
    ax.set_ylabel("")
    ax.set_title("GRS-VAE reference workflow: cell-type-specific route scores")
    fig.tight_layout()
    fig.savefig(args.outdir / "reference_route_scores.png", dpi=180)
    print(f"Wrote demonstration outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
