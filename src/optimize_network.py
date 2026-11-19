import argparse
import contextlib
import gzip
import hashlib
import json
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GPSConv, GCNConv

HERE = Path(__file__).resolve().parent
DATA_BACKGROUND_OFFSET = 10000019
PROJECT = HERE.parent
TASKS = ('NASH', 'NASH-HCC', 'NASH-to-HCC')
DE_FILES = {
    'NASH': 'GSE164760_HealthLiver_NASH.tsv',
    'NASH-HCC': 'GSE164760_HealthLiver_NASH-HCC.tsv',
    'NASH-to-HCC': 'GSE164760_adjNASH-NASH-HCC_NASH-HCC.tsv',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class GPS_MLP_Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=.1):
        super().__init__()
        self.output_dim = output_dim
        self.pre_linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GPSConv(hidden_dim, conv=GCNConv(hidden_dim, hidden_dim),
                             heads=heads, dropout=dropout)
        self.conv2 = GPSConv(hidden_dim, conv=GCNConv(hidden_dim, hidden_dim),
                             heads=heads, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        x = F.relu(self.pre_linear(x))
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        delta = self.mlp(h)
        return (delta + delta.T).mul_(0.5)


def index_ppi(raw):
    genes = np.unique(raw[:, :2])
    uv = np.searchsorted(genes, raw[:, :2])
    edge_index = np.stack([uv.reshape(-1), uv[:, ::-1].reshape(-1)])
    unique_edges = np.unique(edge_index, axis=1)
    return genes, edge_index, unique_edges


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


def de_groups(path, genes, k=1000):
    if k < 1:
        raise ValueError('--data-top-k must be positive')
    logfc = gene_logfc(path, genes)
    valid = np.flatnonzero(np.isfinite(logfc))
    if 2 * k > len(valid):
        raise ValueError(f'--data-top-k={k} needs at least {2*k} valid PPI genes; '
                         f'found {len(valid)}. Maximum K is {len(valid)//2}.')
    order = valid[np.argsort(np.abs(logfc[valid]), kind='stable')]
    return order[-k:], order[:k]


def select_data_seeds(path, genes, kk):
    if kk < 1:
        raise ValueError('--data-seed-top-k must be positive')
    logfc = gene_logfc(path, genes)
    valid = np.flatnonzero(np.isfinite(logfc))
    if kk > len(valid):
        raise ValueError(f'--data-seed-top-k={kk} exceeds {len(valid)} valid PPI genes')
    order = valid[np.argsort(np.abs(logfc[valid]), kind='stable')]
    return order[-kk:]


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
def sample_background(adj, seeds, args, rng):
    if args.degree_backend == 'scipy':
        cpu_csr = sparse.csr_matrix(adj.detach().cpu().numpy())
        degrees = cpu_csr.sum(axis=1).A1.astype(np.int32)
    else:
        degrees = adj.sum(dim=1).to(torch.int32).cpu().numpy()
    return background_indices(degrees, seeds, args.num_reps, args.bin_size, rng,
                              args.background_backend)


def diffuse(adj, seeds, args, rng=None, indices=None):
    n = adj.shape[0]
    if indices is None:
        indices = sample_background(adj, seeds, args, rng)
    y = adj.new_zeros(n)
    y[torch.as_tensor(seeds, device=adj.device)] = 1.0 / len(seeds)
    y_all = adj.new_zeros((len(indices), n))
    y_all.scatter_(1, torch.as_tensor(indices, device=adj.device), 1.0 / len(seeds))
    def propagate(adjacency, y, y_all):
        col_sums = adjacency.sum(dim=0)
        inverse = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums).reciprocal()
        w = adjacency * inverse.unsqueeze(0)
        obs = y.clone()
        for _ in range(args.diffusion_steps):
            obs = args.alpha * (w @ obs) + (1 - args.alpha) * y
        rand = y_all.clone()
        for _ in range(args.diffusion_steps):
            rand = args.alpha * (rand @ w.T) + (1 - args.alpha) * y_all
        return obs, rand

    if args.checkpoint_diffusion and torch.is_grad_enabled() and adj.requires_grad:
        obs, rand = checkpoint(propagate, adj, y, y_all, use_reentrant=False)
    else:
        obs, rand = propagate(adj, y, y_all)
    log_obs = torch.log(obs.clamp(min=1e-12))
    log_rand = torch.log(rand.clamp(min=1e-12))
    if not torch.isfinite(log_obs).all() or not torch.isfinite(log_rand).all():
        raise RuntimeError('Nonfinite diffusion output')
    mean = log_rand.mean(dim=0)
    std = torch.linalg.vector_norm(log_rand - mean, dim=0) / (len(indices) ** .5)
    z = (log_obs - mean) / (std + 1e-6)
    return z, {'observed_heat': obs, 'null_mean': mean, 'null_std': std,
               'random_seed_mass_mean': float(y_all.sum(dim=1).mean())}


@torch.no_grad()
def score_stats(z, details, high, low):
    std = details['null_std']
    quantiles = torch.quantile(std, std.new_tensor([.01, .5]))
    return dict(high_mean_z=float(z[high].double().mean()),
                low_mean_z=float(z[low].double().mean()),
                high_mean_heat=float(details['observed_heat'][high].mean()),
                low_mean_heat=float(details['observed_heat'][low].mean()),
                null_std_min=float(std.min()), null_std_p01=float(quantiles[0]),
                null_std_median=float(quantiles[1]),
                null_std_below_epsilon=int((std < 1e-6).sum()))


DATA_STAT_FIELDS = ['high_mean_z', 'low_mean_z', 'high_mean_heat', 'low_mean_heat',
                    'null_std_min', 'null_std_p01', 'null_std_median', 'null_std_below_epsilon']
DATA_REWARD_STAT_FIELDS = ['data_reward_raw', 'data_reward_asinh',
                           'high_mean_z_asinh', 'low_mean_z_asinh']


def data_reward_from_z(z, high, low, transform='none', z_scale=3.):
    if transform not in ('none', 'asinh-z'):
        raise ValueError(f'Unknown data reward transform: {transform}')
    if not np.isfinite(z_scale) or z_scale <= 0:
        raise ValueError('--data-reward-z-scale must be finite and positive')
    z64 = z.to(torch.float64)
    raw = z64[high].mean() - z64[low].mean()
    high_asinh = z_scale * torch.asinh(z64[high] / z_scale)
    low_asinh = z_scale * torch.asinh(z64[low] / z_scale)
    transformed = high_asinh.mean() - low_asinh.mean()
    stats = dict(data_reward_raw=float(raw.detach()), data_reward_asinh=float(transformed.detach()),
                 high_mean_z_asinh=float(high_asinh.detach().mean()),
                 low_mean_z_asinh=float(low_asinh.detach().mean()))
    return (raw if transform == 'none' else transformed), stats


def score_data_reward(adj, high, low, args, rng=None, indices=None, data_seeds=None):
    if data_seeds is None:
        if args.use_data_reward and getattr(args, 'data_seed_top_k', None) not in (None, len(high)):
            raise ValueError('Explicit data_seeds required when data_seed_top_k differs from reward K')
        data_seeds = high
    stats = {key: None for key in DATA_STAT_FIELDS + DATA_REWARD_STAT_FIELDS +
             ['random_seed_mass_mean', 'data_random_seed_mass_mean']}
    data = adj.new_zeros((), dtype=torch.float64)
    if not args.use_data_reward:
        return data, stats, None
    if not len(data_seeds) or not len(high) or not len(low):
        raise ValueError('Empty data seed or comparison group')
    sampled = indices if indices is not None else sample_background(adj, data_seeds, args, rng)
    z, details = diffuse(adj, data_seeds, args, indices=sampled)
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite Z-score in data diffusion')
    data, reward_stats = data_reward_from_z(z, high, low,
        args.data_reward_transform, args.data_reward_z_scale)
    stats.update(reward_stats)
    stats.update(score_stats(z, details, high, low))
    stats['random_seed_mass_mean'] = details['random_seed_mass_mean']
    stats['data_random_seed_mass_mean'] = details['random_seed_mass_mean']
    return data, stats, sampled


def weighted_reward(data, args):
    return args.data_weight * data if args.use_data_reward else data.new_zeros(())


def gradient_diagnostics(actor, data, structural, args):
    params = list(actor.parameters())
    terms = {'data': args.reward_coefficient * args.data_weight * data,
             'structure': args.structure_weight * structural}
    grads, report = {}, {}
    for name, term in terms.items():
        values = (torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                  if term.requires_grad else [None] * len(params))
        values = [(g.detach().cpu() if getattr(args, 'offload_saved_tensors', False) else g.detach())
                  if g is not None else None for g in values]
        norms = [g.norm().double().square() for g in values if g is not None]
        norm = torch.stack(norms).sum().sqrt() if norms else term.new_zeros(())
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite {name} component gradient')
        report[name + '_gradient_norm'] = float(norm)
        grads[name] = values
    for a, b in [('data', 'structure')]:
        denominator = report[a + '_gradient_norm'] * report[b + '_gradient_norm']
        if denominator:
            dot = sum((ga * gb).sum(dtype=torch.float64) for ga, gb in zip(grads[a], grads[b])
                      if ga is not None and gb is not None)
            report[a + '_' + b + '_gradient_cosine'] = float(dot) / denominator
        else:
            report[a + '_' + b + '_gradient_cosine'] = None
    return report


def make_candidate(previous, raw_delta, scale):
    candidate = torch.clamp(previous.detach() + torch.tanh(raw_delta) * scale, 0, 1)
    return (candidate + candidate.T).mul_(0.5)


def retain_top_k(adj, k, backend='batched', chunk_rows=256, symmetrization='mean'):
    if symmetrization not in ('mean', 'max'):
        raise ValueError('Unknown Top-K symmetrization: ' + symmetrization)
    k = min(int(k), adj.shape[1])
    if k < 1 or chunk_rows < 1:
        raise ValueError('top-k and chunk-rows must be positive')
    with torch.no_grad():
        step = 1 if backend == 'reference' else chunk_rows
        indices = torch.cat([torch.topk(adj[start:start+step], k, dim=1).indices
                             for start in range(0, adj.shape[0], step)], dim=0)
    values = adj.gather(1, indices)
    selected = torch.zeros_like(adj).scatter(1, indices, values)
    if symmetrization == 'max':
        # A one-sided selection retains its full weight; no repeated halving.
        return torch.maximum(selected, selected.T)
    return (selected + selected.T).mul_(0.5)


def _nonzero_stride_overlap(shape, strides):
    span = 1
    for stride, size in sorted((stride, size) for size, stride in zip(shape, strides) if size > 1):
        if stride < span:
            return True
        span += (size - 1) * stride
    return False


def pack_saved_tensor(tensor, pin_memory=False):
    device = tensor.device
    if tensor.layout != torch.strided:
        return device, tensor.detach().cpu(), None, None, False
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    compact = tensor.detach()
    for dim, (size, stride) in enumerate(zip(shape, strides)):
        if size > 1 and stride == 0:
            compact = compact.narrow(dim, 0, 1)
    if compact.numel() and _nonzero_stride_overlap(compact.shape, strides):
        return device, tensor.detach(), shape, strides, True
    pinned = bool(pin_memory and device.type == 'cuda')
    cpu = torch.empty_strided(tuple(compact.shape), strides, dtype=tensor.dtype,
                              device='cpu', pin_memory=pinned)
    cpu.copy_(compact)
    return device, cpu, shape, strides, False


def unpack_saved_tensor(packed):
    device, tensor, shape, strides, passthrough = packed
    if passthrough:
        return tensor
    if strides is None:
        return tensor.to(device)
    if device.type == 'cpu':
        restored = tensor
    else:
        restored = torch.empty_strided(tuple(tensor.shape), strides, dtype=tensor.dtype, device=device)
        restored.copy_(tensor, non_blocking=tensor.is_pinned())
    return restored.as_strided(shape, strides)


def saved_tensor_context(args, device):
    if args.offload_saved_tensors and args.train_actor:
        return torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: pack_saved_tensor(tensor, pin_memory=device.type == 'cuda'),
            unpack_saved_tensor)
    return contextlib.nullcontext()


@torch.no_grad()
def support_changes(previous_edges, current_edges, n):
    before = torch.unique(previous_edges[0] * n + previous_edges[1], sorted=True)
    after = torch.unique(current_edges[0] * n + current_edges[1], sorted=True)
    if not before.numel() or not after.numel():
        common = 0
    else:
        positions = torch.searchsorted(before, after)
        matched = (positions < len(before)) & (before[positions.clamp(max=len(before)-1)] == after)
        common = int(matched.sum())
    return dict(directed_edges_added=len(after)-common, directed_edges_removed=len(before)-common)


@torch.no_grad()
def make_structure_reference(adj, args):
    projected = retain_top_k(adj, args.top_k, args.topk_backend,
                            args.topk_chunk_rows, args.topk_symmetrization)
    edges = torch.nonzero(projected > 0, as_tuple=False).T.contiguous()
    if edges.shape[1] == 0:
        raise ValueError('Initial projected PPI has no reference edges')
    values = projected[edges[0], edges[1]].detach().clone()
    return edges, values


def structure_loss(adj, original_edges, backend='indexed', original_mask=None,
                   mode='original-norm', reference=None):
    if mode == 'reference-mse':
        if reference is None:
            raise ValueError('reference-mse requires a fixed projected-PPI reference')
        edges, target = reference
        return (adj[edges[0], edges[1]] - target).square().mean()
    if mode != 'original-norm':
        raise ValueError('Unknown structure mode: ' + mode)
    if backend == 'reference':
        return torch.norm(adj * original_mask, p=2)
    return torch.norm(adj[original_edges[0], original_edges[1]], p=2)


def save_network(adj, path):
    a = adj.detach().cpu().numpy()
    src, dst = np.nonzero(a)
    np.save(path, np.vstack([src, dst, a[src, dst]]).T)


def validate_learning_rate_options(args):
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('--lr must be positive and finite')
    if args.min_lr is not None and (not np.isfinite(args.min_lr) or args.min_lr < 0):
        raise ValueError('--min-lr must be nonnegative and finite')
    if args.lr_scheduler == 'cosine' and args.min_lr is not None and args.min_lr > args.lr:
        raise ValueError('--min-lr must not exceed --lr for cosine decay')


def make_lr_scheduler(optimizer, args):
    validate_learning_rate_options(args)
    if args.lr_scheduler == 'constant':
        return None
    minimum = args.lr * .01 if args.min_lr is None else args.min_lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 1, 1), eta_min=minimum)


@contextlib.contextmanager
def tee_log(path):
    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log
        def write(self, message):
            self.stream.write(message)
            self.log.write(message)
            self.log.flush()
            return len(message)
        def flush(self):
            self.stream.flush()
            self.log.flush()
    with path.open('w', buffering=1) as f:
        with contextlib.redirect_stdout(Tee(sys.stdout, f)), contextlib.redirect_stderr(Tee(sys.stderr, f)):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=f)
                raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task', choices=TASKS, default='NASH-to-HCC', help='Canonical task names (not legacy disease aliases)')
    p.add_argument('--project-root', type=Path, default=PROJECT)
    p.add_argument('--ppi', type=Path)
    p.add_argument('--features', type=Path)
    p.add_argument('--de-file', type=Path)
    p.add_argument('--output', type=Path, help='New run directory; default timestamped directory')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds', type=int, nargs='+', choices=[5], default=[5],
                   help='Legacy full-data run identifier; kept as 5 for output/RNG compatibility')
    p.add_argument('--epochs', type=int, default=2001)
    p.add_argument('--lr', type=float, default=1e-7)
    p.add_argument('--lr-scheduler', choices=['constant', 'cosine'], default='constant',
                   help='constant: legacy fixed LR; cosine: decay from --lr at epoch 0 to --min-lr at the final epoch')
    p.add_argument('--min-lr', type=float, default=None,
                   help='Cosine final LR, default 0.01 * --lr; must be between 0 and --lr. Ignored for constant scheduling. A one-epoch run uses --lr only')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--data-weight', type=float, default=100.)
    p.add_argument('--use-data-reward', action=argparse.BooleanOptionalAction, default=True,
                   help='Top-KK high-|logFC| seeds for separate data diffusion; disabling skips it')
    p.add_argument('--reward-coefficient', type=float, default=-1., help='Notebook lambda_3')
    p.add_argument('--structure-weight', type=float, default=1.)
    p.add_argument('--structure-mode', choices=['original-norm', 'reference-mse'], default='original-norm',
                   help='original-norm: legacy norm on original PPI edges; reference-mse: mean squared deviation on fixed pre-action projected-PPI edges (no new-edge penalty)')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--topk-symmetrization', choices=['mean', 'max'], default='mean',
                   help='mean: legacy averaging; max: union keeps full weight if either endpoint selects an edge')
    p.add_argument('--scale-start', type=float, default=.001)
    p.add_argument('--scale-end', type=float, default=1.)
    p.add_argument('--scale-epochs', type=int, help='Default epochs, matching notebook scale denominator')
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--diffusion-steps', type=int, default=20)
    p.add_argument('--num-reps', type=int, default=100)
    p.add_argument('--bin-size', type=int, default=100)
    p.add_argument('--data-top-k', type=int, default=1000,
                   help='K genes in EACH high/low |logFC| reward group; independent of seed KK and graph --top-k')
    p.add_argument('--data-seed-top-k', type=int, default=None,
                   help='KK highest-|logFC| genes used ONLY as data diffusion seeds; default: --data-top-k')
    p.add_argument('--data-reward-transform', choices=['none', 'asinh-z'], default='none',
                   help='none: raw mean-Z contrast; asinh-z: mean contrast after per-gene s*asinh(Z/s)')
    p.add_argument('--data-reward-z-scale', type=float, default=3.,
                   help='Positive finite s for per-gene asinh transform; default 3')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--background-seed', type=int, help='Optional reproducible background; omitted matches unseeded notebook')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--diffusion-backend', choices=['dense'], default='dense',
                   help='Dense propagation on projected graph; only selected candidate edges receive gradients')
    p.add_argument('--checkpoint-diffusion', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--offload-saved-tensors', action=argparse.BooleanOptionalAction, default=False,
                   help='Store autograd activations on CPU; reduce GPU memory at the cost of transfers/RAM')
    p.add_argument('--profile-memory', action='store_true',
                   help='Record per-epoch CUDA peak allocated/reserved memory (MiB)')
    p.add_argument('--train-actor', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--background-mode', choices=['resample', 'common'], default='resample',
                   help='common reuses random numbers each epoch, with current degree bins')
    p.add_argument('--diagnostic-every', type=int, default=25,
                   help='0 disables periodic component gradients and post-Top-K evaluations')
    p.add_argument('--max-grad-norm', type=float, default=0., help='0 disables gradient clipping')
    p.add_argument('--degree-backend', choices=['torch', 'scipy'], default='scipy',
                   help='scipy preserves notebook float reduction before integer degree binning; torch avoids CPU transfer')
    p.add_argument('--background-backend', choices=['vectorized', 'reference'], default='vectorized')
    p.add_argument('--topk-backend', choices=['batched', 'reference'], default='batched')
    p.add_argument('--topk-chunk-rows', type=int, default=256)
    p.add_argument('--structure-backend', choices=['indexed', 'reference'], default='indexed')
    p.add_argument('--reference', action='store_true', help='Reference binning/Top-K/structure paths; diffusion stays differentiable')
    p.add_argument('--log-every', type=int, default=1, help='Console interval; CSV always saved every epoch')
    p.add_argument('--profile', action='store_true', help='Synchronize CUDA around timed stages')
    return p


def load_inputs(args):
    root = args.project_root.resolve()
    paths = {
        'ppi': args.ppi or root / 'data/Network/STRING_PPI/STRING.npy.gz',
        'features': args.features or root / 'data/Feature/STRING_Feature_mtx.npy',
        'de': args.de_file or root / 'data/GEO' / DE_FILES[args.task],
    }
    if str(paths['ppi']).endswith('.gz'):
        with gzip.open(paths['ppi'], 'rb') as stream:
            raw = np.load(stream, allow_pickle=False)
    else:
        raw = np.load(paths['ppi'], allow_pickle=False)
    genes, edges, original_edges = index_ppi(raw)
    del raw
    x = np.load(paths['features'], allow_pickle=False).astype(np.float32)
    if x.ndim != 2 or x.shape[0] != len(genes) or not np.isfinite(x).all():
        raise ValueError('Invalid feature shape/alignment/nonfinite features')
    de, nonde = de_groups(paths['de'], genes, args.data_top_k)
    provenance = {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in paths.items()}
    return genes, x, edges, original_edges, de, nonde, provenance


def run(args):
    validate_learning_rate_options(args)
    if not np.isfinite(args.data_reward_z_scale) or args.data_reward_z_scale <= 0:
        raise ValueError('--data-reward-z-scale must be finite and positive')
    if args.data_seed_top_k is None:
        args.data_seed_top_k = args.data_top_k
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this execution session')
    if args.reference:
        args.diffusion_backend = 'dense'
        args.degree_backend = 'scipy'
        args.background_backend = 'reference'
        args.topk_backend = 'reference'
        args.structure_backend = 'reference'
    genes, x_np, edges_np, original_np, de, nonde, provenance = load_inputs(args)
    data_seeds = select_data_seeds(Path(provenance['de']['path']), genes, args.data_seed_top_k)
    n = len(genes)
    if args.top_k > n:
        raise ValueError('--top-k exceeds number of genes')
    reward_mode = 'data_only' if args.use_data_reward else 'structure_only'
    if args.train_actor and reward_mode == 'structure_only' and args.structure_weight == 0:
        raise ValueError('Data reward is disabled and structure weight is zero: no training objective')
    transform_tag = 'raw' if args.data_reward_transform == 'none' else f'asinh_z_s_{args.data_reward_z_scale:g}'
    out = args.output or PROJECT / 'outputs/optimization' / args.task / f'K_{args.data_top_k}' / f'KK_{args.data_seed_top_k}' / reward_mode / transform_tag / f'{args.topk_symmetrization}_{args.structure_mode}' / ('train' if args.train_actor else 'frozen') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    x = torch.tensor(x_np, device=device)
    edges = torch.tensor(edges_np, dtype=torch.long, device=device)
    original = torch.tensor(original_np, dtype=torch.long, device=device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(method='differentiable_projected_data_diffusion', data_reward='mean_z_high_abs_logFC_minus_mean_z_low_abs_logFC',
                  effective_min_lr=(args.lr * .01 if args.min_lr is None else args.min_lr) if args.lr_scheduler == 'cosine' else None,
                  lr_schedule_definition='constant or cosine over epochs 0..epochs-1, independent of scale-epochs; one epoch uses initial LR',
                  lr_logging='learning_rate is the optimizer LR used at that epoch, before scheduler advancement; frozen Actor keeps initial LR and performs no optimizer/scheduler steps',
                  topk_symmetrization_definition=('(selected+selected.T)/2' if args.topk_symmetrization == 'mean' else 'maximum(selected,selected.T)'),
                  structure_definition=('norm(A[original_PPI_edges],2)' if args.structure_mode == 'original-norm' else
                                        'mean((A[reference_edges]-A_ref[reference_edges])**2)'),
                  structure_reference=('none' if args.structure_mode == 'original-norm' else
                                       'original binary PPI projected once before any Actor action, using selected top-k and symmetrization; fixed for entire fold'),
                  structure_new_edge_penalty=False,
                  data_reward_definition=('mean(Z_high)-mean(Z_low)' if args.data_reward_transform == 'none' else
                                          'mean(s*asinh(Z_high/s))-mean(s*asinh(Z_low/s))'),
                  data_reward_log_semantics='data_reward is the selected optimization reward; raw/asinh versions logged separately; high_mean_z/low_mean_z stay raw',
                  reward_mode=reward_mode,
                  data_seed_definition='highest data_seed_top_k absolute logFC genes; independent of reward groups',
                  data_seed_genes=len(data_seeds),
                  data_seed_overlap_high=int(np.isin(data_seeds, de).sum()),
                  data_seed_overlap_low=int(np.isin(data_seeds, nonde).sum()),
                  data_background_offset=DATA_BACKGROUND_OFFSET,
                  memory_optimization='normalization inside diffusion checkpoint; fresh-sum in-place scaling; optional stride-preserving CPU activation storage',
                  activation_offload='preserve original strides; collapse broadcast dimensions; overlapping-view passthrough',
                  disabled_reward_logging='null/blank; branch diffusion skipped; weighted contribution zero',
                  unprefixed_diffusion_stats='data diffusion',
                  gradient_scope='observed and null diffusion; conditional on sampled indices; one graph update',
                  zero_std_subgradient='zero via centered vector norm; denominator remains std + 1e-6',
                  control_semantics='no-train-actor skips backward/optimizer; train-mode buffers/dropout retained',
                  data_group_selection='finite first-probe logFC; no P-value filter; disjoint extremes',
                  data_group_ties='ascending abs(logFC), then ascending PPI node index',
                  reward_stage='after_topk', structure_stage='after_topk', network_stage='last_scored_graph',
                  projection_gradient='selected values only; discrete membership; no straight-through estimator',
                  random_background='with replacement; duplicate indices overwrite (notebook semantics)',
                  n_genes=n, n_features=x.shape[1], initial_encoder_edges=edges.shape[1],
                  original_binary_edges=original.shape[1], high_abs_logfc_genes=len(de), low_abs_logfc_genes=len(nonde),
                  python=platform.python_version(), torch=str(torch.__version__),
                  numpy=np.__version__, script_sha256=sha256(__file__), inputs=provenance)
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    pd.DataFrame({'node_index': np.arange(n), 'gene': genes}).to_csv(out / 'gene_index.csv', index=False)
    groups = pd.DataFrame({'node_index': np.arange(n), 'gene': genes, 'group': 'not_selected'})
    groups['logFC'] = gene_logfc(Path(provenance['de']['path']), genes)
    groups['abs_logFC'] = groups['logFC'].abs()
    groups.loc[~np.isfinite(groups['logFC']), 'group'] = 'missing_or_invalid'
    groups.loc[de, 'group'] = 'high_abs_logFC'
    groups.loc[nonde, 'group'] = 'low_abs_logFC'
    groups['is_data_seed'] = np.isin(np.arange(n), data_seeds)
    groups.to_csv(out / 'data_reward_groups.csv', index=False)
    data_rng = np.random.default_rng(None if args.background_seed is None else
                                     args.background_seed + DATA_BACKGROUND_OFFSET)
    def tick():
        if args.profile and device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()
    summaries = []
    with tee_log(out / 'console.log'):
        print(f'Task={args.task}; method=differentiable_projected_data_diffusion; topk_symmetrization={args.topk_symmetrization}; structure_mode={args.structure_mode}; rewards={reward_mode}; data_reward_transform={args.data_reward_transform}; z_scale={args.data_reward_z_scale:g}; reward_stage=after_topk; device={device}; data_top_k={args.data_top_k}; data_seed_top_k={args.data_seed_top_k}; high_abs_logFC={len(de)}; low_abs_logFC={len(nonde)}', flush=True)
        print(f'Output: {out.resolve()}', flush=True)
        print(f'Learning rate: scheduler={args.lr_scheduler}; initial={args.lr:g}; '
              f'minimum={config["effective_min_lr"]}; epochs={args.epochs}; train_actor={args.train_actor}', flush=True)
        for fold in args.folds:
            folder = out / f'fold_{fold}'
            folder.mkdir()
            (folder / 'data_selection.json').write_text(json.dumps({'data_seed_nodes': data_seeds.tolist(),
                'data_high_nodes': de.tolist(), 'data_low_nodes': nonde.tolist(),
                'use_data_reward': args.use_data_reward, 'full_fit': True}, indent=2))
            pd.DataFrame({'node_index': data_seeds, 'gene': genes[data_seeds]}).to_csv(folder / 'data_seed_genes.csv', index=False)
            (folder / 'background_rng_start.json').write_text(json.dumps(
                {'data': data_rng.bit_generator.state}, indent=2))
            actor = GPS_MLP_Actor(x.shape[1], args.hidden, n, args.heads, args.dropout).to(device)
            optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
            scheduler = make_lr_scheduler(optimizer, args) if args.train_actor else None
            adj = x.new_zeros((n, n))
            adj[original[0], original[1]] = 1.0
            mask = adj.clone() if args.structure_mode == 'original-norm' and args.structure_backend == 'reference' else None
            structure_reference = None
            if args.structure_mode == 'reference-mse':
                structure_reference = make_structure_reference(adj, args)
                ref_edges, ref_values = structure_reference
                np.save(folder / 'structure_reference_edges.npy',
                        np.column_stack((ref_edges.detach().cpu().numpy().T,
                                         ref_values.detach().cpu().numpy())))
                print(f'[Fold {fold}] Fixed structure reference: {ref_values.numel()} directed edges; initial projected PPI', flush=True)
            current_edges = edges
            diagnostic_fields = ['data_gradient_norm', 'structure_gradient_norm',
                                 'data_structure_gradient_cosine', 'post_topk_data_same_background',
                                 'post_topk_reward_same_background', 'scored_graph_data_recheck_error',
                                 'directed_edges_added', 'directed_edges_removed',
                                 'post_topk_data_independent_background']
            diagnostic_fields += ['post_topk_data_raw_same_background', 'post_topk_data_asinh_same_background',
                                  'post_topk_data_raw_independent_background', 'post_topk_data_asinh_independent_background']
            fields = ['task', 'fold', 'epoch', 'total_reward', 'data_reward',
                      'structure_loss', 'l1_reg', 'total_loss', 'directed_edges_after_topk',
                      'scale', 'diffusion_backend', 'random_seed_mass_mean', 'gradient_norm',
                      'train_actor', 'high_mean_z', 'low_mean_z', 'high_mean_heat', 'low_mean_heat',
                      'null_std_min', 'null_std_p01', 'null_std_median', 'null_std_below_epsilon',
                      'actor_seconds', 'diffusion_seconds', 'backward_seconds', 'topk_seconds',
                      'diagnostic_seconds', 'epoch_seconds', 'use_data_reward',
                      'data_weighted_reward', 'data_random_seed_mass_mean',
                      'data_seed_count', 'data_top_k', 'data_seed_top_k',
                      'data_reward_transform', 'data_reward_z_scale'] + DATA_REWARD_STAT_FIELDS + diagnostic_fields
            fields += ['cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
                       'topk_symmetrization', 'structure_mode', 'structure_reference_edge_count',
                       'structure_weighted_loss']
            fields += ['learning_rate']
            common_seed = (args.background_seed if args.background_seed is not None else args.seed) + fold
            evaluation_seed = args.seed + 1000003 + fold
            (folder / 'diagnostic_sampling.json').write_text(json.dumps(dict(
                common_seed=common_seed, independent_evaluation_seed=evaluation_seed,
                data_common_seed=common_seed+DATA_BACKGROUND_OFFSET,
                data_independent_evaluation_seed=evaluation_seed+DATA_BACKGROUND_OFFSET,
                note='Matched data seed sets; evaluation does not advance training RNG. Current-graph degree bins are recomputed.'), indent=2))
            with (folder / 'reward_history.csv').open('w', newline='') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for epoch in range(args.epochs):
                    learning_rate = float(optimizer.param_groups[0]['lr'])
                    optimizer.zero_grad(set_to_none=True)
                    if args.profile_memory and device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    start = tick()
                    actor.train()  # no-train control retains exactly the same dropout mechanism
                    diagnose = args.diagnostic_every > 0 and (epoch % args.diagnostic_every == 0 or epoch == args.epochs-1)
                    diagnostic = {}
                    with torch.set_grad_enabled(args.train_actor), saved_tensor_context(args, device):
                        scale = args.scale_start + (args.scale_end-args.scale_start)*epoch/(args.scale_epochs or args.epochs)
                        adj = make_candidate(adj, actor(x, current_edges), scale)
                        after_actor = tick()
                        adj = retain_top_k(adj, args.top_k, args.topk_backend, args.topk_chunk_rows,
                                           args.topk_symmetrization)
                        after_projection = tick()
                        if args.background_mode == 'common':
                            step_rng = np.random.default_rng(common_seed+DATA_BACKGROUND_OFFSET)
                        else:
                            step_rng = data_rng
                        data_tensor, stats, indices = score_data_reward(
                            adj, de, nonde, args, rng=step_rng, data_seeds=data_seeds)
                        reward_tensor = weighted_reward(data_tensor, args)
                        after_diffusion = tick()
                        structural = structure_loss(adj, original, args.structure_backend, mask,
                                                    args.structure_mode, structure_reference)
                        l1 = float(torch.norm(adj.detach(), p=1))
                        loss = args.reward_coefficient*reward_tensor + args.structure_weight*structural
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'Nonfinite loss at fold {fold}, epoch {epoch}')
                        loss_value, structural_value = float(loss.detach()), float(structural.detach())
                        reward = float(reward_tensor.detach())
                        data_reward = float(data_tensor.detach()) if args.use_data_reward else None
                        optimizer.zero_grad(set_to_none=True)
                        grad_norm = 0.
                        if args.train_actor:
                            if args.use_data_reward and not data_tensor.requires_grad:
                                raise RuntimeError('Data reward is disconnected from autograd')
                            if diagnose:
                                diagnostic.update(gradient_diagnostics(actor, data_tensor, structural, args))
                            loss.backward()
                            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                                actor.parameters(), args.max_grad_norm if args.max_grad_norm > 0 else float('inf'),
                                error_if_nonfinite=True))
                            optimizer.step()
                            if scheduler is not None and epoch < args.epochs - 1:
                                scheduler.step()
                        after_backward = tick()
                    # Carry exactly the scored projected graph; NEVER apply Top-K a second time.
                    del data_tensor, reward_tensor, loss, structural
                    adj = adj.detach()
                    previous_edges = current_edges
                    current_edges = torch.nonzero(adj > 0, as_tuple=False).T.contiguous()
                    after_topk = tick()
                    if diagnose:
                        with torch.no_grad():
                            # Recheck the exact scored graph with the same indices; difference should be ~0.
                            rp, replay_stats, _ = score_data_reward(adj, de, nonde, args,
                                indices=indices, data_seeds=data_seeds)
                            diagnostic.update(post_topk_data_same_background=float(rp) if args.use_data_reward else None,
                                post_topk_reward_same_background=float(weighted_reward(rp, args)),
                                scored_graph_data_recheck_error=float(rp)-data_reward if args.use_data_reward else None)
                            diagnostic.update(post_topk_data_raw_same_background=replay_stats['data_reward_raw'],
                                              post_topk_data_asinh_same_background=replay_stats['data_reward_asinh'])
                            diagnostic.update(support_changes(previous_edges, current_edges, n))
                            if args.use_data_reward and not np.isclose(float(rp), data_reward, rtol=1e-4, atol=1e-5):
                                raise RuntimeError('Scored and carried graph rewards differ on the same background')
                            re, evaluation_stats, _ = score_data_reward(adj, de, nonde, args,
                                rng=np.random.default_rng(evaluation_seed+DATA_BACKGROUND_OFFSET),
                                data_seeds=data_seeds)
                            diagnostic.update(post_topk_data_independent_background=float(re) if args.use_data_reward else None)
                            diagnostic.update(post_topk_data_raw_independent_background=evaluation_stats['data_reward_raw'],
                                              post_topk_data_asinh_independent_background=evaluation_stats['data_reward_asinh'])
                            del rp, re
                    end = tick()
                    peak_allocated = (torch.cuda.max_memory_allocated(device) / 2**20
                                      if args.profile_memory and device.type == 'cuda' else None)
                    peak_reserved = (torch.cuda.max_memory_reserved(device) / 2**20
                                     if args.profile_memory and device.type == 'cuda' else None)
                    row = dict(task=args.task, fold=fold, epoch=epoch, total_reward=reward,
                        data_reward=data_reward, structure_loss=structural_value,
                        l1_reg=l1, total_loss=loss_value, directed_edges_after_topk=current_edges.shape[1],
                        scale=scale, diffusion_backend='dense',
                        use_data_reward=args.use_data_reward,
                        data_weighted_reward=args.data_weight*data_reward if args.use_data_reward else 0.,
                        data_seed_count=len(data_seeds) if args.use_data_reward else 0,
                        data_top_k=args.data_top_k, data_seed_top_k=args.data_seed_top_k,
                        data_reward_transform=args.data_reward_transform, data_reward_z_scale=args.data_reward_z_scale,
                        cuda_peak_allocated_mib=peak_allocated, cuda_peak_reserved_mib=peak_reserved,
                        topk_symmetrization=args.topk_symmetrization, structure_mode=args.structure_mode,
                        structure_reference_edge_count=(structure_reference[1].numel() if structure_reference is not None else None),
                        structure_weighted_loss=args.structure_weight*structural_value,
                        learning_rate=learning_rate,
                        gradient_norm=grad_norm, train_actor=args.train_actor, **stats, **diagnostic,
                        actor_seconds=after_actor-start, diffusion_seconds=after_diffusion-after_projection,
                        backward_seconds=after_backward-after_diffusion, topk_seconds=(after_projection-after_actor)+(after_topk-after_backward),
                        diagnostic_seconds=end-after_topk, epoch_seconds=end-start)
                    writer.writerow(row)
                    f.flush()
                    if epoch % args.log_every == 0 or epoch == args.epochs-1:
                        data_text = f'{data_reward:.6f}' if args.use_data_reward else 'disabled'
                        raw_text = f'{stats["data_reward_raw"]:.6f}' if args.use_data_reward else 'disabled'
                        print(f'[Fold {fold}] Epoch {epoch:04d} Reward(post-TopK)={reward:.6f} '
                              f'Data={data_text} DataRaw={raw_text} Structure={structural_value:.6f} Loss={loss_value:.6f} '
                              f'Grad={grad_norm:.6g} LR={learning_rate:.6g} Edges={current_edges.shape[1]} Seconds={end-start:.3f}', flush=True)
                        if peak_allocated is not None:
                            print(f'CUDA peak: allocated={peak_allocated:.1f} MiB; reserved={peak_reserved:.1f} MiB', flush=True)
                    if diagnose:
                        print('Diagnostics: ' + json.dumps(diagnostic), flush=True)
            network_file = folder / f'adj_edges_{args.task}_fold_{fold}.npy'
            save_network(adj, network_file)
            torch.save({'actor_state_dict': {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'fold': fold,
                        'lr_scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'last_epoch': epoch, 'network_file': network_file.name}, folder / 'actor.pt')
            summary = dict(fold=fold, epochs_completed=args.epochs, final_reward_after_topk=reward,
                           final_data_reward=data_reward,
                           final_data_reward_raw=stats['data_reward_raw'], final_data_reward_asinh=stats['data_reward_asinh'],
                           data_reward_transform=args.data_reward_transform, data_reward_z_scale=args.data_reward_z_scale,
                           topk_symmetrization=args.topk_symmetrization, structure_mode=args.structure_mode,
                           structure_reference_file=('structure_reference_edges.npy' if structure_reference is not None else None),
                           lr_scheduler=args.lr_scheduler, final_learning_rate=learning_rate,
                           final_loss=loss_value, saved_directed_edges=current_edges.shape[1],
                           network_file=network_file.name)
            (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            (out / 'summary.json').write_text(json.dumps({'config': config, 'completed_folds': summaries}, indent=2))
    return out


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate_learning_rate_options(args)
    except ValueError as error:
        p.error(str(error))
    if (args.epochs < 1 or args.hidden < 1 or args.heads < 1 or args.hidden % args.heads
        or args.lr <= 0 or args.top_k < 1 or args.num_reps < 2 or args.bin_size < 1
        or args.diffusion_steps < 1 or args.threads < 1 or args.topk_chunk_rows < 1
        or args.log_every < 1 or not 0 <= args.alpha <= 1 or not 0 <= args.dropout < 1
        or args.data_top_k < 1 or args.diagnostic_every < 0 or args.max_grad_norm < 0
        or (args.data_seed_top_k is not None and args.data_seed_top_k < 1)
        or not np.isfinite(args.data_reward_z_scale) or args.data_reward_z_scale <= 0
        or any(f != 5 for f in args.folds) or len(set(args.folds)) != len(args.folds)
        or (args.scale_epochs is not None and args.scale_epochs < 1)):
        p.error('Invalid hyperparameters or folds')
    run(args)


if __name__ == '__main__':
    main()
