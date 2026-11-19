import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATA_BACKGROUND_OFFSET = 10000019


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gene_logfc(path, genes):
    frame = pd.read_csv(path, sep='\t', usecols=['Gene.Symbol', 'logFC'])
    frame = frame.loc[frame['Gene.Symbol'].notna()].copy()
    frame['gene'] = frame['Gene.Symbol'].astype(str).str.split('///')
    frame = frame.explode('gene')
    frame['gene'] = frame.gene.str.strip()
    frame = frame.loc[~frame.gene.isin(['', '---'])].copy()
    frame['logFC'] = pd.to_numeric(frame['logFC'], errors='coerce')
    return (frame.drop_duplicates('gene', keep='first').set_index('gene')['logFC']
            .reindex(genes).to_numpy(float))


def select_seeds(logfc, count):
    valid = np.flatnonzero(np.isfinite(logfc))
    if count > len(valid):
        raise ValueError(f'Need {count} mapped genes with finite logFC; found {len(valid)}')
    return valid[np.argsort(np.abs(logfc[valid]), kind='stable')][-count:]


def degree_bins(degrees, bin_size):
    groups = {}
    for node, degree in enumerate(degrees):
        groups.setdefault(degree, []).append(node)
    sorted_degrees = sorted(groups)
    bins, i = [], 0
    while i < len(sorted_degrees):
        val = groups[sorted_degrees[i]]
        start = sorted_degrees[i]
        while len(val) < bin_size:
            i += 1
            if i >= len(sorted_degrees):
                break
            val.extend(groups[sorted_degrees[i]])
        end = sorted_degrees[i] if i < len(sorted_degrees) else start
        if len(val) < bin_size and bins:
            bins[-1] = (bins[-1][0], end, bins[-1][2] + val)
        else:
            bins.append((start, end, val))
        i += 1
    return [np.asarray(nodes, dtype=np.int64) for _, _, nodes in bins]


def background_indices(degrees, seeds, reps, bin_size, rng, backend='vectorized'):
    pools = degree_bins(degrees, bin_size)
    membership = np.empty(len(degrees), dtype=np.int64)
    for i, nodes in enumerate(pools):
        membership[nodes] = i
    seed_bins = membership[seeds]
    if backend == 'reference':
        return np.array([[rng.choice(pools[b]) for b in seed_bins] for _ in range(reps)],
                        dtype=np.int64)
    sizes = np.array([len(pools[b]) for b in seed_bins], dtype=np.int64)
    offsets = rng.integers(0, sizes[None, :], size=(reps, len(seeds)), dtype=np.int64)
    indices = np.empty_like(offsets)
    for b in np.unique(seed_bins):
        columns = np.flatnonzero(seed_bins == b)
        indices[:, columns] = pools[b][offsets[:, columns]]
    return indices


@torch.no_grad()
def diffuse(adj, seeds, indices, alpha, steps):
    n = adj.shape[0]
    y = adj.new_zeros(n)
    y[torch.as_tensor(seeds, device=adj.device)] = 1.0 / len(seeds)
    y_all = adj.new_zeros((len(indices), n))
    y_all.scatter_(1, torch.as_tensor(indices, device=adj.device), 1.0 / len(seeds))
    col_sums = adj.sum(dim=0)
    inverse = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums).reciprocal()
    w = adj * inverse.unsqueeze(0)
    obs = y.clone()
    for _ in range(steps):
        obs = alpha * (w @ obs) + (1 - alpha) * y
    rand = y_all.clone()
    for _ in range(steps):
        rand = alpha * (rand @ w.T) + (1 - alpha) * y_all
    log_obs = torch.log(obs.clamp(min=1e-12))
    log_rand = torch.log(rand.clamp(min=1e-12))
    if not torch.isfinite(log_obs).all() or not torch.isfinite(log_rand).all():
        raise RuntimeError('Nonfinite diffusion output')
    mean = log_rand.mean(dim=0)
    std = torch.linalg.vector_norm(log_rand - mean, dim=0) / (len(indices) ** .5)
    z = (log_obs - mean) / (std + 1e-6)
    if not torch.isfinite(z).all():
        raise RuntimeError('Nonfinite Z-score')
    return z, dict(null_std_min=float(std.min()),
                   null_std_median=float(std.median()),
                   null_std_below_epsilon=int((std < 1e-6).sum()),
                   random_seed_mass_mean=float(y_all.sum(dim=1).mean()))
