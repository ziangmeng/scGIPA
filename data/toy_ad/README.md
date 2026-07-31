# Synthetic toy AD inputs

This directory contains the fixed synthetic inputs used by `examples/run_toy_ad_demo.py`.

- `expression_counts.csv`: 324 synthetic cells by 48 synthetic genes.
- `metadata.csv`: broad cell type, disease state and ordered stage label.
- `route_prior_matrix.csv`: per-cell route-prior vector supplied to GRS-VAE.
- `route_prior_evidence.csv`: compact route evidence table with synthetic GWAS/eQTL P values.

All genes, cells, P values and effects are simulated. These files are intentionally small and are included solely to make the public workflow inspectable and reproducible.
