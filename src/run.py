import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GPSConv, GATConv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import re
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    make_scorer
)

torch.manual_seed(42)
np.random.seed(42)

disease = 'NASH-HCC'

if disease == 'NASH-HCC':
    df_risk_genes_STRING = pd.read_csv('../data/NASH-HCC_HC_genes.csv')

lst_risk_genes_STRING = df_risk_genes_STRING['Genes'].tolist()

df_ngtv_pool = pd.read_csv('../data/ngtv_pool.csv')
lst_ngtv_pool = df_ngtv_pool["Genes"].tolist()

ary_edgelist = np.load('../data/STRING.npy')


feature_mtx = np.load("../data/Feature_mtx.npy")

feature_torch = torch.tensor(feature_mtx, dtype=torch.float)
print(feature_torch.shape)

print(len(feature_mtx))
print(len(feature_mtx[0]))

nodelist = []
for x in ary_edgelist:
    nodelist.append(x[0])
    nodelist.append(x[1])
nodelist = list(set(nodelist))
nodelist = sorted(nodelist)

dict_symbol2idx = {}
dict_idx2symbol = {}

for i in range(len(nodelist)):
    tmp_symbol = nodelist[i]
    dict_symbol2idx[tmp_symbol] = i
    dict_idx2symbol[i] = tmp_symbol

lst_edge_0 = []
lst_edge_1 = []

for x in ary_edgelist:
    lst_edge_0.append(dict_symbol2idx[x[0]])
    lst_edge_1.append(dict_symbol2idx[x[1]])
    
    lst_edge_0.append(dict_symbol2idx[x[1]])
    lst_edge_1.append(dict_symbol2idx[x[0]])
    
edge_index_STRING = torch.tensor([lst_edge_0, lst_edge_1], dtype=torch.long)


adj_STRING = torch.zeros(len(nodelist), len(nodelist))
adj_STRING[edge_index_STRING[0], edge_index_STRING[1]] = 1


class GPS_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_hidden, output_dim, heads=4, dropout=0.1):
        super(GPS_MLP, self).__init__()
        from torch_geometric.nn import GCNConv, GPSConv
        import torch.nn as nn

        self.pre_linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GPSConv(channels=hidden_dim, conv=GCNConv(hidden_dim, hidden_dim),
                             heads=heads, dropout=dropout)
        self.conv2 = GPSConv(channels=hidden_dim, conv=GCNConv(hidden_dim, hidden_dim),
                             heads=heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_dim)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.pre_linear(x))
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        out = self.mlp(h)
        return out


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

feature_torch = feature_torch.to(device)
ary_edgelist_optimized = np.load('../data/ary_edgelist_optimized.npy')
edge_index_opt = torch.tensor(
    np.array([[int(x[0]), int(x[1])] for x in ary_edgelist_optimized]).T,
    dtype=torch.long
).to(device)
data = Data(x=feature_torch, edge_index=edge_index_opt)

model_path = "../data/model.pt"

model = GPS_MLP(
    input_dim=data.num_node_features,
    hidden_dim=512,
    mlp_hidden=64,
    output_dim=2,
    heads=4,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()

gene_names = np.array(list(dict_symbol2idx.keys())) 
order = np.argsort(-prob)
ranked_genes = gene_names[order]

df = pd.DataFrame({"Gene": ranked_genes})
df.to_csv("Gene_Ranking.csv", index=False, encoding="utf-8")
