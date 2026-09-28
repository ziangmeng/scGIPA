# Study workflow and reference implementation

The synthetic workflow demonstrates model training and paired interventions.
Synthetic outputs are not estimates from the study cohort.

## Model correspondence

The implementation follows the expression encoder, cell embedding, signed
expression-weighted route encoder, integrated decoder and four-state classifier
used by scGIPA. Disease labels are targets of the classifier and auxiliary stage
loss; they are not inputs to the encoder, decoder or classifier.

The auxiliary stage target is 0, 1 and 2 for HA, PCI and AD. YA contributes to
reconstruction and four-state classification but not auxiliary stage regression.
The total loss adds reconstruction and classification losses, 0.2 times the
stage loss, 0.001 times KL divergence and 0.2 times directional anchor loss.
Anchor targets are donor-balanced expression differences in the training set.

| Setting | CPU demonstration | Manuscript analysis |
| --- | --- | --- |
| Input | 864 synthetic cells, 24 simulated donors, 64 genes | Study-specific expression and genetic inputs |
| States | YA, HA, PCI, AD | YA, HA, PCI, AD |
| Hidden / latent dimensions | 64 / 12 | 384 / 48 |
| Training epochs | 40 by default | 25 per initialization |
| Learning rate | 0.002 | 0.0002 |
| Partitions | One fixed train / validation / test partition | Five recorded donor partitions and three initializations each |
| Evaluation output | Synthetic test-donor predictions | Study analyses use their recorded donor partitions and evaluation definitions |

The public runner keeps the best validation-loss checkpoint and evaluates its
separate test donors afterwards. It does not reconstruct manuscript AUCs from
the synthetic example. Checkpoint selection and evaluation roles are recorded
in each output manifest.

## Applying the code to expression data

Provide the CSV contracts described in the main README. For large sparse
datasets, perform count aggregation and data loading with appropriate sparse
tools rather than exporting the full matrix to CSV. The small public runner
holds its input in memory and is intended to demonstrate the computational
steps. The same model and intervention functions can be called from a study
data loader.

Within a donor and cell type, sum counts for pseudobulk analysis, normalize to
10,000 counts and apply log1p. The runner requires at least 10 cells per
donor–cell profile. I uses the preceding-stage training-donor mean, with equal
weight per reference donor, and averages effects over target test donors.
M preserves expression and its encoding and masks the selected route input.
Joint-set interventions operate on all selected features together; their
effects need not equal sums of single-gene effects.

## Preparing a study genetic prior

The model consumes a cell-type by gene matrix of signed weights. The study
constructs this matrix from AD GWAS association, MAGMA gene and cell-property
results, and cell-type-specific single-cell cis-eQTL evidence from SingleBrain, with allele harmonization and LD matching. Direct matching uses the same variant; proxy matching uses r² ≥ 0.8. Both GWAS and eQTL associations meet P ≤ 10⁻³. BrainMeta cortical cis-eQTLs supplement cell types lacking SingleBrain coverage.
Signed gene scores are weighted by evidence source, MAGMA gene Z and positive
cell-property coefficients. Source weights are 1.0 for direct cell-type eQTL,
0.8 for LD-proxy cell-type eQTL and 0.4 for bulk-cortex evidence. At repeated
gene–cell coordinates the largest absolute weighted value is retained. Within
each cell type, nonzero weights are divided by their 95th absolute percentile
and clipped to [-4, 4].

Full upstream GWAS processing, MAGMA, signed-LD matching and eQTL harmonization
require the original resources and their study-specific QC. They are not run
by the synthetic demonstration. The bundled prior is generated synthetically;
it is not a substitute for a harmonized study prior.

## Source resources

- Hippocampal transcriptomes: GSE268609; DOI 10.1038/s41586-026-10169-4.
- AD GWAS: GWAS Catalog accession GCST90704646.
- Cell-type regulatory evidence: SingleBrain single-cell cis-eQTLs and BrainMeta cortical cis-eQTLs, as specified in the accompanying manuscript.
- Functional enrichment: g:Profiler; module membership and analysis criteria
  are described in the manuscript methods.

Obtain source datasets and reference panels from their providers under the
applicable access terms. This repository does not redistribute them.
