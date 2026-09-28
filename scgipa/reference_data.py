"""Synthetic inputs and validated donor-aware data contracts."""
from pathlib import Path
import numpy as np
import pandas as pd

CELL_TYPES = ["Microglia", "Astrocytes", "Excitatory_neurons"]
STATES = ["YA", "HA", "PCI", "AD"]
CONTRASTS = [(0, 1), (1, 2), (2, 3)]

def normalise_counts(counts):
    counts = np.asarray(counts, dtype=np.float32)
    return np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e4)

def make_reference_dataset(directory, seed=7):
    """Write 864 synthetic cells from 24 artificial donors with 64 genes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    genes = [f"GENE_{i:03d}" for i in range(1, 65)]
    prior = np.zeros((3, 64), dtype=np.float32)
    for c in range(3):
        prior[c, c * 6:c * 6 + 6] = [1, -.8, .7, -.6, .5, -.4]
    baseline = rng.normal(1.2, .3, 64)
    records, counts = [], []
    for s, state in enumerate(STATES):
        for donor in range(6):
            donor_id = f"SIM_{state}_{donor + 1:02d}"
            donor_effect = rng.normal(0, .18, 64)
            split = ["train", "train", "train", "validation", "test", "test"][donor]
            for c, cell in enumerate(CELL_TYPES):
                programme = np.zeros(64)
                programme[c * 8:c * 8 + 8] = .45
                state_effect = np.zeros(64)
                state_effect[24:32] = s * .25
                state_effect[32:40] = -.15 * s
                trajectory = [[0, .5, -.2, .7], [0, -.3, .45, .8], [0, .2, .7, .35]][c][s]
                state_effect += trajectory * prior[c]
                for _ in range(12):
                    cell_id = f"SIM_CELL_{len(records) + 1:04d}"
                    lam = np.exp(np.clip(baseline + donor_effect + programme + state_effect + rng.normal(0, .15, 64), -2, 4))
                    counts.append(rng.poisson(lam))
                    records.append(dict(cell_id=cell_id, donor_id=donor_id, cell_type=cell, state=state, split=split))
    pd.DataFrame(counts, index=[r['cell_id'] for r in records], columns=genes).rename_axis('cell_id').to_csv(directory/'expression_counts.csv')
    pd.DataFrame(records).to_csv(directory/'metadata.csv', index=False)
    pd.DataFrame(prior, index=CELL_TYPES, columns=genes).rename_axis('cell_type').to_csv(directory/'route_prior_matrix.csv')
    evidence = [dict(cell_type=CELL_TYPES[c], gene=genes[g], route_weight=float(prior[c,g]), source='synthetic')
                for c,g in zip(*np.nonzero(prior))]
    pd.DataFrame(evidence).to_csv(directory/'route_prior_evidence.csv', index=False)

def load_reference_dataset(directory):
    directory = Path(directory)
    counts = pd.read_csv(directory/'expression_counts.csv', index_col='cell_id')
    metadata = pd.read_csv(directory/'metadata.csv', dtype=str).set_index('cell_id')
    prior = pd.read_csv(directory/'route_prior_matrix.csv', index_col='cell_type')
    if not counts.index.is_unique or not metadata.index.is_unique or not counts.columns.is_unique:
        raise ValueError('cell IDs and gene names must be unique')
    if set(metadata.index) != set(counts.index):
        raise ValueError('expression and metadata cell IDs must match exactly')
    metadata = metadata.loc[counts.index]
    required = ['donor_id', 'cell_type', 'state', 'split']
    if metadata[required].isna().any().any():
        raise ValueError('metadata fields cannot be missing')
    if not set(metadata.state).issubset(STATES) or set(metadata.split) != {'train','validation','test'}:
        raise ValueError('use YA/HA/PCI/AD and train/validation/test labels')
    if (metadata.groupby('donor_id')[['state','split']].nunique() != 1).any().any():
        raise ValueError('a donor must belong to exactly one state and one split')
    cells = sorted(metadata.cell_type.unique())
    if (set(prior.index) != set(cells) or set(prior.columns) != set(counts.columns)
            or not prior.index.is_unique or not prior.columns.is_unique):
        raise ValueError('prior gene and cell-type axes must match the input')
    prior = prior.loc[cells, counts.columns]
    if (not np.isfinite(counts.to_numpy()).all() or (counts.to_numpy() < 0).any()
            or not np.isfinite(prior.to_numpy()).all()):
        raise ValueError('counts must be finite and nonnegative; prior must be finite')
    if not np.allclose(counts.to_numpy(), np.round(counts.to_numpy())):
        raise ValueError('supply counts, not normalized expression')
    for split in ['train','validation','test']:
        if set(metadata.loc[metadata.split==split,'state']) != set(STATES):
            raise ValueError(f'{split} must contain donors from all four states')
    return counts.to_numpy(np.float32), metadata.reset_index(), prior.to_numpy(np.float32), list(counts.columns), cells

def pseudobulk(counts, metadata, min_cells=10):
    profiles, records = [], []
    for (donor, cell, state, split), frame in metadata.groupby(['donor_id','cell_type','state','split'], sort=True):
        if len(frame) < min_cells:
            continue
        profiles.append(counts[frame.index.to_numpy()].sum(0))
        records.append(dict(donor_id=donor,cell_type=cell,state=state,split=split,n_cells=len(frame)))
    if not profiles:
        raise ValueError('no donor-cell profiles meet the minimum cell count')
    return normalise_counts(np.stack(profiles)), pd.DataFrame(records)
