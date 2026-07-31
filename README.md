# GRS-VAE

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

**GRS-VAE** is a route-prior variational framework for integrating disease-GWAS evidence, regulatory-QTL evidence and single-cell gene expression. Rather than ranking genes only at the locus level, GRS-VAE represents each candidate as a **cell-type-specific gene route** and asks how that route contributes to a disease-state expression contrast in the learned latent model.

This repository is a complete, lightweight **reference demonstration** of the v7.10 analytical design. It provides a reproducible synthetic AD-like benchmark, an implementation of the conditional VAE and a route-masking analysis. It does not distribute real participant-level expression, GWAS, eQTL, genotypes, model weights or restricted reference panels.

## Contents

- [Overview](#overview)
- [Conceptual framework](#conceptual-framework)
- [Reference demonstration](#reference-demonstration)
- [Installation](#installation)
- [Running the workflow](#running-the-workflow)
- [Input data contract](#input-data-contract)
- [Model architecture and objectives](#model-architecture-and-objectives)
- [Route-masking analysis](#route-masking-analysis)
- [Output files](#output-files)
- [Adapting the workflow to a study](#adapting-the-workflow-to-a-study)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Overview

Genome-wide association studies identify disease-associated loci, but a locus can contain many correlated variants and several plausible target genes. Regulatory-QTL studies connect variants to gene regulation, whereas single-cell transcriptomics reveals the cell populations and disease states in which those genes are active. GRS-VAE integrates these complementary layers in one conditional generative model.

The framework has three goals:

1. Build a gene-level regulatory prior from harmonized GWAS and eQTL evidence within a relevant cell type.
2. Learn an expression latent space that retains cell identity and disease-state structure while incorporating that prior.
3. Score each gene-cell route by masking its prior contribution and measuring the modeled change along a selected disease contrast.

The output is a ranked set of routes rather than a single global gene list. This makes it possible to distinguish, for example, a microglial candidate from the same gene acting in an astrocytic context.

## Conceptual framework

```text
Disease GWAS                Regulatory QTL                 Single-cell RNA
     │                            │                               │
     └──── variant harmonization ─┴──── cell-type route prior ──────┤
                                      │                              │
                                      ▼                              ▼
                         cell type + disease state + expression
                                      │
                                      ▼
                             conditional GRS-VAE
                                      │
                         route present versus route masked
                                      │
                                      ▼
                     cell-type-specific gene route ranking
```

For cell type `c`, gene `g` and matched variant `v`, a study-scale route prior can combine harmonized effect directions, association strength and locus-level colocalization. The reference implementation stores the resulting values in a per-cell route-prior vector. The public benchmark uses simulated values so that every intermediate file can be inspected.

## Reference demonstration

The bundled synthetic benchmark is small enough to run on CPU and is designed to demonstrate the complete data flow without external downloads. It contains:

- **324 cells**: 36 cells in each combination of three broad cell types and three disease states.
- **Three cell classes**: Microglia, Astrocytes and Excitatory neurons.
- **Three ordered states**: Control, MCI and AD.
- **48 synthetic genes** and **15 cell-type-specific routes**.
- A count-like expression matrix, cell metadata, a route-prior matrix and a compact route-evidence table.

The synthetic data-generating process includes cell-class programs and route-associated disease-state shifts. Therefore, route scoring has a known cell-context structure while remaining entirely independent of real human data.

## Installation

Clone the repository and create a clean Python environment:

```bash
git clone https://github.com/ziangmeng/GRS-VAE.git
cd GRS-VAE
python -m venv .venv
```

Activate the environment and install the required packages:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reference workflow requires Python 3.10+ and PyTorch 2.1+. It is intentionally CPU-compatible; no GPU, R installation or external genomic reference panel is required for the demonstration.

## Running the workflow

Execute the end-to-end AD reference demonstration:

```bash
python examples/run_reference_ad_demo.py --epochs 80
```

To write results to a custom directory:

```bash
python examples/run_reference_ad_demo.py \
    --epochs 120 \
    --seed 7 \
    --outdir outputs/ad_reference_run
```

The script loads the fixed benchmark stored in `data/reference_ad/`, trains the conditional model and exports route-level results. Adding `--regenerate` recreates the synthetic input matrices from the chosen seed before training.

## Input data contract

The reference files define a compact input contract that can be adapted to larger studies.

### Expression matrix

`expression_counts.csv` has one row per cell and one column per gene. The first column, `cell_id`, is a stable cell identifier. The example uses count-like values; the workflow applies `log1p` internally before training.

### Cell metadata

`metadata.csv` contains `cell_id`, `cell_type`, `state` and `stage`. `cell_type` indexes the learned cell-identity embedding. `state` indexes the categorical disease-state embedding. `stage` is a numeric ordering used by the auxiliary stage-regression objective.

### Route-prior matrix

`route_prior_matrix.csv` has the same row order and gene columns as the expression matrix. A non-zero element indicates that the gene has route-prior support in that cell's annotated cellular context. In a study-scale analysis, this matrix is assembled from cell-type-matched GWAS-QTL evidence rather than assigned synthetically.

### Route-evidence table

`route_prior_evidence.csv` is a transparent long-format companion table. It records `cell_type`, `gene`, `route_weight`, `synthetic_gwas_p` and `synthetic_eqtl_p`. The P values are simulated and illustrate data structure only; they are not biological association statistics.

## Model architecture and objectives

For a cell with log-transformed expression vector `x`, cell type `c`, disease state `s` and route-prior vector `r`, the encoder estimates a Gaussian latent representation:

```text
q(z | x) = Normal(mu(x), diag(sigma(x)^2))
```

The decoder receives both the sampled latent state and a context vector:

```text
h(c, s, r) = E_cell(c) + E_state(s) + W_route r
x_hat = Decoder([z, h(c, s, r)])
```

`E_cell` and `E_state` are learned embeddings, while `W_route` projects the route prior into the latent context. In the reference model, optimization combines four complementary objectives:

```text
L = L_reconstruction + 0.001 L_KL + 0.35 L_state + 0.15 L_stage
```

- `L_reconstruction` preserves the observed single-cell expression profile.
- `L_KL` regularizes the variational latent distribution.
- `L_state` predicts the categorical disease state from the expression latent representation.
- `L_stage` predicts the ordered disease-stage value.

This construction gives the model access to expression structure, cellular context, disease-state context and the genetically informed route prior at the same time.

## Route-masking analysis

After training, the model encodes each cell once and decodes it under two matched contexts:

```text
x_hat_full   = Decoder([z, h(c, s, r)])
x_hat_masked = Decoder([z, h(c, s, 0)])
delta_route  = x_hat_full - x_hat_masked
```

For each cell type, the reference workflow estimates a Control-to-AD expression direction from the observed data. A route score is obtained by projecting the route-dependent reconstruction difference onto that gene-specific disease contrast. Positive and negative scores therefore describe how the modeled route contribution aligns with the selected contrast within its annotated cell class.

## Output files

The default run writes `outputs/reference_ad_demo/`:

```text
reference_cell_metadata.csv        Input metadata copied for traceability
reference_route_prior.csv          Input evidence table copied for traceability
reference_route_scores.csv         Per-gene, per-cell-type masking score
reference_celltype_summary.csv     Mean, maximum and number of routes by cell type
reference_route_scores.png         Ranked route-score visualization
```

`reference_route_scores.csv` is the central output. Each row contains the gene, its cellular context, input route weight, route-masking score and mean expression in the Control and AD benchmark states.

## Adapting the workflow to a study

The public benchmark is intentionally compact, but the code maps directly to a study-scale workflow. Replace the synthetic matrices with a harmonized single-cell expression matrix and metadata, then construct a route-prior matrix using a consistent reference genome, allele-aware GWAS-QTL matching and cell-type-specific regulatory evidence. The same conditional model and masking procedure can then be evaluated for contrasts appropriate to the study design, such as healthy-to-disease or early-to-late progression.

For real analyses, `grs_vae/model.py` supplies the reusable conditional VAE module and `examples/run_reference_ad_demo.py` provides the minimal training and scoring template. Genome-build harmonization, locus definition, regional colocalization and cohort-specific quality control are study-specific preprocessing stages and should be recorded alongside each analysis.

## Repository layout

```text
GRS-VAE/
├── data/reference_ad/             Fixed synthetic reference inputs
├── examples/
│   └── run_reference_ad_demo.py   End-to-end training and route scoring
├── grs_vae/
│   ├── model.py                   Conditional route-prior VAE
│   └── reference_data.py          Synthetic benchmark generation and loading
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite the accompanying GRS-VAE manuscript when available. The framework builds on variational autoencoding and single-cell latent-variable modeling concepts, including *Auto-Encoding Variational Bayes* and *Deep generative modeling for single-cell transcriptomics*.
