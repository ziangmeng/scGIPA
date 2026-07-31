# GRS-VAE

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

**GRS-VAE** is a route-prior variational framework for connecting disease-associated variants, regulatory evidence, cell identity and disease-state-associated single-cell expression. The central analytical unit is a **cell-type-specific gene route**: a gene is considered together with its regulatory support and the cellular context in which that support is observed.

This repository provides a lightweight, fully runnable public demonstration of the core v7.10 concepts. It contains **only synthetic data**; no individual-level single-cell profiles, GWAS summary statistics, eQTL summary statistics or protected reference files are distributed here.

## Contents

- [What the model represents](#what-the-model-represents)
- [Quick start](#quick-start)
- [Toy workflow](#toy-workflow)
- [Outputs](#outputs)
- [Using your own data](#using-your-own-data)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## What the model represents

The workflow combines four quantities for each cell:

1. **Expression state**: the observed gene-expression profile.
2. **Cell identity**: the broad cell class or a harmonized cell annotation.
3. **Disease state**: an ordered or categorical phenotype label, such as control, prodromal and disease.
4. **Route prior**: a gene-level vector constructed from matched disease-GWAS and regulatory-QTL evidence within a cell type.

GRS-VAE learns a compact latent representation of expression and decodes it with the cell identity, disease state and route prior. A route-masking analysis then compares reconstructed expression with the route prior present versus masked. The resulting score ranks candidate gene-cell routes by their modeled contribution to a selected disease-state contrast.

```text
GWAS variants + regulatory-QTL evidence
                │
                ▼
       cell-type-specific route prior
                │
single-cell expression ──► GRS-VAE ◄── cell type + disease state
                │                         │
                └──────── route masking ──┘
                              │
                              ▼
                 gene-cell route prioritization
```

## Quick start

Create a Python environment and install the small public dependency set:

```bash
git clone https://github.com/ziangmeng/GRS-VAE.git
cd GRS-VAE
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete synthetic example:

```bash
python examples/run_toy_ad_demo.py --epochs 80
```

The example runs on CPU in well under a minute on a typical laptop.

## Toy workflow

`examples/run_toy_ad_demo.py` reads a bundled 324-cell synthetic dataset across three broad cell types (`Microglia`, `Astrocytes`, `Excitatory_neurons`) and three states (`Control`, `MCI`, `AD`). It includes 48 synthetic genes and 15 deliberately constructed cell-type-specific routes. The data-generating mechanism makes route-associated genes shift across the disease states so that the end-to-end workflow can be inspected without external resources. Use `--regenerate` to recreate the inputs from the fixed random seed.

The script performs the following steps:

1. Generate synthetic raw-count-like expression, metadata and a route-prior matrix.
2. Transform counts with `log1p` and encode cell type and disease state.
3. Train a conditional VAE with reconstruction, KL, state-classification and stage-regression objectives.
4. Decode each cell with its route prior and with that prior masked.
5. Project the reconstructed difference onto the Control-to-AD expression contrast within each cell type.
6. Export per-route scores, per-cell-type summaries and a compact ranking figure.

## Outputs

The command writes a small `outputs/toy_ad_demo/` directory:

```text
toy_cell_metadata.csv       Synthetic cell annotation
toy_route_prior.csv         Synthetic route-prior evidence table
toy_route_scores.csv        Gene-cell route masking scores
toy_celltype_summary.csv    Per-cell-type route summary
toy_route_scores.png        Ranked route-score figure
```

The numerical values are intentionally synthetic and are for software demonstration only.

## Using your own data

The public toy workflow is deliberately compact. A study-scale analysis replaces the synthetic inputs with:

- a normalized or count-based single-cell expression matrix;
- harmonized cell-type and disease-state annotations;
- a cell-type-specific route prior assembled from matched GWAS and eQTL evidence;
- locus-level colocalization and QC outputs used to prioritize biologically coherent routes.

The `GRSVAE` module in `grs_vae/model.py` contains the reusable conditional VAE components. The route-prior construction, genome-build harmonization, colocalization and cohort-specific QC must be tailored to each study and are intentionally not bundled with protected or large source data.

## Repository layout

```text
GRS-VAE/
├── grs_vae/
│   ├── model.py                 Conditional route-prior VAE
│   └── toy_data.py              Small synthetic data generator
├── examples/
│   └── run_toy_ad_demo.py       End-to-end public demonstration
├── data/toy_ad/                 324-cell synthetic input tables
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite the accompanying GRS-VAE manuscript when available. The framework builds on variational autoencoding and single-cell latent-variable modeling concepts, including *Auto-Encoding Variational Bayes* and *Deep generative modeling for single-cell transcriptomics*.
