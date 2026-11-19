import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from propagation import (DATA_BACKGROUND_OFFSET, background_indices, diffuse,
                         gene_logfc, select_seeds, sha256)

ROOT = Path(__file__).resolve().parents[1]
DE_FILES = {
    'NASH': 'GSE164760_HealthLiver_NASH.tsv',
    'NASH-HCC': 'GSE164760_HealthLiver_NASH-HCC.tsv',
    'NASH-to-HCC': 'GSE164760_adjNASH-NASH-HCC_NASH-HCC.tsv',
}


def load_genes(path):
    frame = pd.read_csv(path, keep_default_na=False)
    column = 'gene_symbol' if 'gene_symbol' in frame else 'gene'
    if 'node_index' not in frame or column not in frame:
        raise ValueError('Gene index needs node_index and gene_symbol (or gene) columns')
    frame = frame.sort_values('node_index')
    if (not np.array_equal(frame.node_index, np.arange(len(frame)))
            or not frame[column].is_unique or frame[column].eq('').any()):
        raise ValueError('Expected unique genes and contiguous zero-based node indices')
    return frame[column].to_numpy(dtype=str)


def load_graph(path, n):
    if path.suffix == '.npz':
        graph = sparse.load_npz(path).tocsr().astype(np.float32)
        graph.sum_duplicates()
        graph.eliminate_zeros()
        graph.sort_indices()
    elif path.suffix == '.npy':
        edges = np.load(path, allow_pickle=False)
        if edges.ndim != 2 or edges.shape[1] != 3 or not np.isfinite(edges).all():
            raise ValueError('NPY network must contain numeric (source, target, weight) rows')
        uv = edges[:, :2].astype(np.int64)
        if (not np.array_equal(uv, edges[:, :2]) or (uv < 0).any()
                or (uv >= n).any() or (edges[:, 2] <= 0).any()):
            raise ValueError('Invalid network indices or nonpositive weights')
        graph = sparse.csr_matrix((edges[:, 2].astype(np.float32), (uv[:, 0], uv[:, 1])),
                                  shape=(n, n))
        if graph.nnz != len(edges):
            raise ValueError('Duplicate directed edge records')
    else:
        raise ValueError('Use a SciPy sparse .npz or numeric directed-edge .npy file')
    if (graph.shape != (n, n) or not np.isfinite(graph.data).all()
            or (graph.data < 0).any()
            or (graph - graph.T).nnz):
        raise ValueError('Expected a symmetric, nonnegative network in gene-index order')
    return graph


def run_task(task, args):
    started = time.perf_counter()
    mapping = args.gene_index
    network = args.network or ROOT / 'networks' / task / 'network.npz'
    de_path = args.de_file or ROOT / 'data/GEO' / DE_FILES[task]
    names = load_genes(mapping)
    graph = load_graph(network, len(names))
    logfc = gene_logfc(de_path, names)
    if args.seed_file:
        seed_frame = pd.read_csv(args.seed_file, keep_default_na=False)
        if 'diffusion_seed_order' in seed_frame:
            seed_frame = seed_frame.sort_values('diffusion_seed_order')
        seeds = seed_frame.node_index.to_numpy(dtype=np.int64)
        symbol_column = 'gene_symbol' if 'gene_symbol' in seed_frame else 'gene'
        if (len(seeds) < 1 or len(np.unique(seeds)) != len(seeds)
                or np.any(seeds < 0) or np.any(seeds >= len(names))):
            raise ValueError('Invalid or duplicate seed indices')
        np.testing.assert_array_equal(names[seeds], seed_frame[symbol_column])
    else:
        seeds = select_seeds(logfc, args.seed_top_k)
    degrees = graph.sum(axis=1).A1.astype(np.int32)
    rng_seed = args.background_seed + args.background_fold + DATA_BACKGROUND_OFFSET
    if args.background_indices:
        indices = np.load(args.background_indices, allow_pickle=False)
        if (indices.shape != (args.num_reps, len(seeds))
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(names))):
            raise ValueError('Background indices must have shape (num_reps, seed_count), with valid node IDs')
    else:
        indices = background_indices(degrees, seeds, args.num_reps, args.bin_size,
                                     np.random.default_rng(rng_seed), args.background_backend)
    print(f'{task}: {len(names)} genes, {len(seeds)} seeds, {graph.nnz} directed edges; '
          f'propagation on {args.device}', flush=True)
    adjacency = torch.from_numpy(graph.toarray()).to(args.device)
    z, diagnostics = diffuse(adjacency, seeds, indices, args.alpha, args.diffusion_steps)
    values = z.cpu().numpy().astype(np.float64)
    ranking = pd.DataFrame({'gene_symbol': names, 'zscore': values}).sort_values(
        'zscore', ascending=False, kind='stable')
    folder = args.output / task
    folder.mkdir(parents=True, exist_ok=True)
    ranking_path = folder / 'gene_zscores.csv'
    ranking.to_csv(ranking_path, index=False, float_format='%.17g')
    pd.DataFrame({'node_index': seeds, 'gene_symbol': names[seeds], 'logFC': logfc[seeds],
                  'abs_logFC': np.abs(logfc[seeds]),
                  'diffusion_seed_order': np.arange(len(seeds))}).to_csv(
                      folder / 'seed_genes.csv', index=False, float_format='%.17g')
    np.save(folder / 'background_indices.npy', indices, allow_pickle=False)
    params = {key: getattr(args, key) for key in ['alpha', 'diffusion_steps', 'num_reps',
        'bin_size', 'background_seed', 'background_fold', 'background_backend']}
    inputs = {name: {'path': str(path.resolve()), 'sha256': sha256(path)}
              for name, path in [('network', network), ('gene_index', mapping), ('de', de_path)]}
    for key in ['seed_file', 'background_indices']:
        if getattr(args, key):
            path = getattr(args, key)
            inputs[key] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    metadata = dict(task=task, parameters=params, rng_seed=rng_seed, seed_count=len(seeds),
        seed_definition=('provided seed-file order' if args.seed_file else 'Top absolute first-probe logFC'),
        inputs=inputs, nodes=len(names), directed_edges=int(graph.nnz),
        undirected_edges=int((graph.nnz + np.count_nonzero(graph.diagonal())) // 2),
        self_loops=int(np.count_nonzero(graph.diagonal())), raw_z=True, all_genes_including_seeds=True,
        extra_topk=False, dtype='float32', device=args.device, threads=args.threads,
        diagnostics=diagnostics, ranking_sha256=sha256(ranking_path),
        versions={'python': platform.python_version(), 'numpy': np.__version__,
                  'torch': str(torch.__version__), 'pandas': pd.__version__},
        seconds=round(time.perf_counter() - started, 3))
    (folder / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{task}: saved {ranking_path}', flush=True)
    del adjacency, graph, z
    gc.collect()
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', '--diseases', nargs='+', choices=list(DE_FILES), default=list(DE_FILES))
    p.add_argument('--network', type=Path, help='Override optimized network; select one task')
    p.add_argument('--gene-index', type=Path, default=ROOT / 'data/gene_index.csv')
    p.add_argument('--de-file', type=Path, help='Override differential-expression TSV; select one task')
    p.add_argument('--seed-file', type=Path, help='Use explicit seed CSV in diffusion_seed_order')
    p.add_argument('--background-indices', type=Path, help='Replay explicit random-seed index array')
    p.add_argument('--seed-top-k', type=int, default=100)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--diffusion-steps', type=int, default=20)
    p.add_argument('--num-reps', type=int, default=100)
    p.add_argument('--bin-size', type=int, default=100)
    p.add_argument('--background-seed', type=int, default=42)
    p.add_argument('--background-fold', type=int, default=5)
    p.add_argument('--background-backend', choices=['vectorized', 'reference'], default='vectorized')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/predictions')
    args = p.parse_args()
    args.tasks = list(dict.fromkeys(args.tasks))
    if len(args.tasks) != 1 and any([args.network, args.de_file, args.seed_file, args.background_indices]):
        p.error('Explicit network/DE/seed/background inputs require exactly one --tasks value')
    if (args.seed_top_k < 1 or args.num_reps < 2 or args.bin_size < 1
            or args.threads < 1 or args.diffusion_steps < 1 or args.background_seed < 0
            or args.background_fold < 0 or not np.isfinite(args.alpha) or not 0 <= args.alpha < 1):
        p.error('Invalid propagation parameters')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    summaries = [run_task(task, args) for task in args.tasks]
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
