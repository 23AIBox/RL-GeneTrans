def retain_top_k(adj_new, K):
    row, col = adj_new.shape
    for i in range(row):
        values, indices = torch.topk(adj_new[i], K) 
        adj_new[i] = torch.zeros_like(adj_new[i]) 
        adj_new[i, indices] = values 
    adj_new = (adj_new + adj_new.T) / 2
    return adj_new

def topk_sparse_delta(delta, k):
    flat = delta.view(-1)
    if k >= flat.numel():
        return delta
    topk_vals, topk_idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[topk_idx] = 1.0
    sparse_delta = delta * mask.view_as(delta)
    return sparse_delta

def save_adj_edges(adj_tensor, filename_prefix="opt_edges"):
    adj_np = adj_tensor.cpu().detach().numpy()
    src, dst = np.nonzero(adj_np)
    weights = adj_np[src, dst]
    edges_array = np.vstack([src, dst, weights]).T
    filename = f"{filename_prefix}.npy"
    np.save(filename, edges_array)
    print(f"Saved adjacency edges to {filename}")
    
def adj_to_edge_index(adj_tensor, threshold=1e-4):
    adj_np = adj_tensor.detach().cpu().numpy()
    src, dst = np.where(adj_np > threshold)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(adj_tensor.device)
    return edge_index