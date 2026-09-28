# scGIPA

[![Reference workflow](https://github.com/ziangmeng/scGIPA/actions/workflows/reference.yml/badge.svg)](https://github.com/ziangmeng/scGIPA/actions/workflows/reference.yml)
![Version](https://img.shields.io/badge/version-0.3.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)

Code demonstration for **Cellular reconfiguration of Alzheimer’s disease genetic susceptibility across ageing and pathology**.

This repository implements **single-cell Genetically Informed Perturbation Analysis (scGIPA)**. It compares four brain states—young adulthood (YA), healthy ageing (HA), preclinical intermediate pathology (PCI) and Alzheimer disease (AD)—and separates two intervention readouts:

- **I, expression-state contribution:** replace selected expression values with an earlier-state reference and recompute the model.
- **M, genetic-route dependence:** keep expression fixed and mask selected genetically anchored inputs.

The repository includes a small synthetic training example, donor-level intervention analysis and the framework figure. Full expression datasets, GWAS/eQTL collections, LD panels and study model weights are obtained separately.

## Framework

![Figure 1. scGIPA framework integrating expression, genetic anchoring and complementary counterfactual interventions.](docs/figures/fig1.png)

**Figure 1.** Genetic associations and brain eQTL evidence define a signed gene–cell prior. Expression and cell identity are integrated with the expression-weighted route signal. Expression replacement and route masking are evaluated separately. The ordinal stage head is auxiliary; I and M use margins from the integrated four-state classifier.

## Quick start

Use Python 3.10 or later in a clean environment:

~~~bash
git clone https://github.com/ziangmeng/scGIPA.git
cd scGIPA
python -m venv .venv
~~~

Activate it with **source .venv/bin/activate** on macOS/Linux or **.venv\Scripts\Activate.ps1** in Windows PowerShell, then run:

~~~bash
python -m pip install -e .
python examples/run_reference_ad_demo.py --epochs 40
~~~

The default demo runs on CPU without data downloads. It uses **864 synthetic cells, 24 artificial donors, three cell types, four states and 64 synthetic genes**. Donors are assigned to disjoint training, validation and test sets. Training uses only training cells; checkpoint selection uses validation cells; final predictions and interventions use test-donor pseudobulk profiles.

~~~bash
# A short execution check
python examples/run_reference_ad_demo.py --epochs 3 --outdir outputs/smoke

# New synthetic inputs are written under this output directory
python examples/run_reference_ad_demo.py --regenerate --seed 11 --outdir outputs/seed11

# Run data-contract and intervention tests
python -m unittest discover -s tests -v
~~~

The synthetic workflow demonstrates computation; it does not reproduce the manuscript's clinical performance or biological results.

## What the model computes

For expression x and cell type c, the route signal is x multiplied elementwise by P[c], where P is the signed genetic prior. The study prior links AD GWAS signals to genes using cell-type-specific single-cell cis-eQTL evidence from SingleBrain, with BrainMeta cortical cis-eQTL evidence supplementing cell types without SingleBrain coverage. The prior is prepared before model training; M is then calculated by masking its expression-weighted inputs. The expression latent, cell embedding and encoded route signal are added and passed to the reconstruction decoder and four-state classifier. **Disease-state labels are supervised targets, not model inputs.**

For an adjacent comparison from state a to state b, define the classifier margin:

~~~text
margin = logit(b) - logit(a)
I = mean over target donors [original margin - expression-reset margin]
M = mean over target donors [original margin - route-masked margin]
~~~

Positive I or M means the original configuration supports the later-state score; negative values mean it offsets that score. These signs do not by themselves mean expression up/down regulation or a causal risk/protective effect.

For I, the reference is the preceding-state mean of **training-donor** pseudobulk expression within the same cell type. Both the expression encoder and expression-dependent route signal are recomputed. For M, the original expression encoding remains fixed while selected route-input coordinates are set to zero. The code also supports joint gene-set interventions.

## Inputs

The CSV runner accepts **--data-dir PATH** with three required files:

| File | Rows and required fields |
| --- | --- |
| expression_counts.csv | One row per cell; unique cell_id, followed by count columns with unique gene names |
| metadata.csv | cell_id, donor_id, cell_type, state, split |
| route_prior_matrix.csv | One row per cell type; cell_type, followed by the same gene names as the expression matrix |

State labels are YA, HA, PCI, AD. Split labels are train, validation, test. A donor must belong to one state and one split; all four states must be represented in each split. The loader aligns metadata and prior axes by identifiers and rejects mismatches, missing values, negative counts and donor overlap. The bundled route_prior_evidence.csv additionally records the synthetic prior's supported routes.

Nuclear counts are normalized to 10,000 and log1p-transformed for training. Interventions first sum raw counts within donor and cell type, then apply the same normalization. Donor–cell profiles with fewer than ten cells are excluded.

The runner holds CSV inputs in memory. For a study-scale sparse matrix, reuse the model and intervention functions with an appropriate sparse data loader. See [study workflow and model correspondence](docs/study_workflow.md) for prior construction, study parameters and required external resources.

## Outputs

The default destination is outputs/reference_ad_demo/, excluded from version control.

| File | Contents |
| --- | --- |
| donor_splits.csv | Donor membership in training, validation and test sets |
| training_history.csv | Training and validation loss at each epoch |
| test_donor_cell_predictions.csv | Four-state probabilities for test-donor pseudobulk profiles |
| test_donor_predictions.csv | Donor predictions after averaging probabilities across represented cells |
| interventions_by_donor.csv | Paired I and M for every gene and a supported-route gene set |
| interventions_summary.csv | Donor-averaged effects, with target-donor counts |
| model.pt | Locally generated parameters and ordered feature labels |
| run_manifest.json | Seed, selected epoch, dimensions, input hashes, software versions and synthetic test accuracy |

## Repository layout

~~~text
scgipa/                   Model, paired interventions and input handling
grs_vae/                  Compatibility imports for existing scripts
examples/                 Synthetic training and intervention workflow
data/reference_ad/        Small, explicitly synthetic inputs
docs/figures/fig1.png      Manuscript framework figure
docs/study_workflow.md    Correspondence to the study analysis
tests/                    Scientific invariants and input validation
tools/check_release.py    File-size and excluded-data checks
~~~

## Version and citation

Version 0.3 names the implementation **scGIPA**, updates the framework figure and manuscript title, and adds an installable Python package. The model architecture, I/M definitions and synthetic inputs are unchanged from v0.2. Existing `grs_vae.EGRDM` and `grs_vae.model.egrdm_loss` imports remain supported; new code should use `scgipa.ScGIPA` and `scgipa.scgipa_loss`. v0.2 state-dictionary checkpoints remain compatible. See [the changelog](CHANGELOG.md).

The reference workflow was checked with Python 3.11, PyTorch 2.3.1, NumPy 1.24.3 and pandas 2.1.4. Small floating-point differences can occur across platforms.

When using this implementation, cite this repository with its commit identifier and the accompanying manuscript, **Cellular reconfiguration of Alzheimer’s disease genetic susceptibility across ageing and pathology**. A publication citation will be added when available. Code is distributed under the [MIT license](LICENSE).
