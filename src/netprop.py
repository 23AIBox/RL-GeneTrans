import numpy as np
import torch
import time
from scipy import sparse
from collections import defaultdict

def run_heat_diffusion_torch(
    adj_matrix,
    seed_nodes,
    num_reps=100,
    alpha=0.5,
    num_its=20,
    bin_size=100,
    degrees=None,
    use_cuda=True
):

    start_time = time.time()

    print(f"Using device: {device}")

    if sparse.issparse(adj_matrix):
        A = torch.tensor(adj_matrix.toarray(), dtype=torch.float32, device=device)
    else:
        A = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
    
    N = A.shape[0]
    seed_nodes = list(seed_nodes)

    col_sums = A.sum(dim=0)
    col_sums[col_sums == 0] = 1.0
    D_inv = 1.0 / col_sums
    W = A * D_inv.unsqueeze(0)

    Y = torch.zeros(N, device=device)
    Y[seed_nodes] = 1.0 / len(seed_nodes)

    F_obs = Y.clone()
    for _ in range(num_its):
        F_obs = alpha * (W @ F_obs) + (1 - alpha) * Y

    if degrees is None:
        if sparse.issparse(adj_matrix):
            degrees = torch.tensor(adj_matrix.sum(axis=1).A1, dtype=torch.int32)
        else:
            degrees = A.sum(dim=1).to(torch.int32)
    else:
        degrees = torch.tensor(degrees, dtype=torch.int32)

    degrees_np = degrees.cpu().numpy()

    def get_bins(degrees, bin_size):
        deg_to_idx = {}
        for i, d in enumerate(degrees):
            deg_to_idx.setdefault(d, []).append(i)
        sorted_degs = sorted(deg_to_idx.keys())
        bins, i = [], 0
        while i < len(sorted_degs):
            val = deg_to_idx[sorted_degs[i]]
            start = sorted_degs[i]
            while len(val) < bin_size:
                i += 1
                if i >= len(sorted_degs): break
                val.extend(deg_to_idx[sorted_degs[i]])
            end = sorted_degs[i] if i < len(sorted_degs) else start
            if len(val) < bin_size and bins:
                bins[-1] = (bins[-1][0], end, bins[-1][2] + val)
            else:
                bins.append((start, end, val))
            i += 1
        return bins

    bins = get_bins(degrees_np, bin_size)
    bin_map = {i: b_idx for b_idx, (_, _, idxs) in enumerate(bins) for i in idxs}
    bin_id_to_nodes = defaultdict(list)
    for idx, bin_id in bin_map.items():
        bin_id_to_nodes[bin_id].append(idx)

    rng = np.random.default_rng()
    Y_all = torch.zeros((num_reps, N), device=device)
    for r in range(num_reps):
        rand_seeds = []
        for i in seed_nodes:
            bin_id = bin_map.get(i)
            candidates = bin_id_to_nodes.get(bin_id, [])
            if candidates:
                rand_choice = rng.choice(candidates)
                rand_seeds.append(rand_choice)
        mask = torch.zeros(N, device=device)
        mask[rand_seeds] = 1.0 / len(rand_seeds)
        Y_all[r] = mask

    F_rand_all = Y_all.clone()
    W_T = W.T
    for _ in range(num_its):
        F_rand_all = alpha * (F_rand_all @ W_T) + (1 - alpha) * Y_all

    F_obs_safe = torch.clamp(F_obs, min=1e-12)
    log_F_obs = torch.log(F_obs_safe)
    F_rand_safe = torch.clamp(F_rand_all, min=1e-12)
    log_F_rand = torch.log(F_rand_safe)

    mask = ~torch.isnan(log_F_rand)
    masked_sum = torch.sum(torch.where(mask, log_F_rand, torch.zeros_like(log_F_rand)), dim=0)
    masked_count = torch.sum(mask, dim=0).clamp(min=1) 
    mean_rand = masked_sum / masked_count

    squared_diff = torch.where(mask, (log_F_rand - mean_rand)**2, torch.zeros_like(log_F_rand))
    std_rand = torch.sqrt(torch.sum(squared_diff, dim=0) / masked_count)
    
    z_scores = (log_F_obs - mean_rand) / (std_rand + 1e-6)
    z_scores_np = z_scores.cpu().numpy()

    significant_nodes = np.where(z_scores_np > 2)[0]

    elapsed = time.time() - start_time
    print(f"\n Time Costing netprop and cal zscore: {elapsed:.2f} s")

    return z_scores_np, F_rand_all.cpu().numpy(), significant_nodes
