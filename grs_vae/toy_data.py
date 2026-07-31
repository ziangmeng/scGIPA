"""A deliberately small synthetic AD-like dataset; no participant data are included."""

from pathlib import Path

import numpy as np
import pandas as pd


CELL_TYPES = ["Microglia", "Astrocytes", "Excitatory_neurons"]
STATES = ["Control", "MCI", "AD"]


def make_toy_dataset(seed=7, cells_per_group=36, n_genes=48):
    """Create log-count-like expression, labels and a cell-type-specific route prior."""
    rng = np.random.default_rng(seed)
    gene_names = np.array([f"GENE_{i:02d}" for i in range(1, n_genes + 1)])
    records, expression, priors = [], [], []

    for cell_id, cell_type in enumerate(CELL_TYPES):
        for state_id, state in enumerate(STATES):
            for _ in range(cells_per_group):
                route = np.zeros(n_genes, dtype=np.float32)
                route_start = cell_id * 5
                route[route_start : route_start + 5] = np.array([1.00, 0.85, 0.70, 0.55, 0.40])
                baseline = 0.55 + rng.normal(0, 0.10, n_genes)
                cell_program = np.zeros(n_genes)
                cell_program[cell_id * 8 : cell_id * 8 + 8] = 0.38
                disease_program = route * (state_id / 2.0) * (0.85 + 0.10 * cell_id)
                lam = np.exp(np.clip(baseline + cell_program + disease_program, -2, 3))
                expression.append(rng.poisson(lam).astype(np.float32))
                records.append({"cell_type": cell_type, "state": state, "stage": state_id / 2.0})
                priors.append(route)

    metadata = pd.DataFrame(records)
    prior_rows = []
    for cell_id, cell_type in enumerate(CELL_TYPES):
        for offset, weight in enumerate([1.00, 0.85, 0.70, 0.55, 0.40]):
            gene = gene_names[cell_id * 5 + offset]
            prior_rows.append({
                "cell_type": cell_type,
                "gene": gene,
                "route_weight": weight,
                "toy_gwas_p": 10 ** (-(3.2 + offset / 3)),
                "toy_eqtl_p": 10 ** (-(4.0 + offset / 4)),
            })

    return np.vstack(expression), metadata, np.vstack(priors), gene_names, pd.DataFrame(prior_rows)


def load_toy_dataset(directory):
    """Load the fixed, tiny synthetic inputs distributed with the repository."""
    directory = Path(directory)
    expression = pd.read_csv(directory / "expression_counts.csv")
    prior_matrix = pd.read_csv(directory / "route_prior_matrix.csv")
    metadata = pd.read_csv(directory / "metadata.csv")
    prior_table = pd.read_csv(directory / "route_prior_evidence.csv")
    gene_names = expression.columns.drop("cell_id").to_numpy()
    return (
        expression.drop(columns="cell_id").to_numpy(dtype=np.float32),
        metadata.drop(columns="cell_id"),
        prior_matrix.drop(columns="cell_id").to_numpy(dtype=np.float32),
        gene_names,
        prior_table,
    )
