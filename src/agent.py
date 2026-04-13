import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GCNConv

class GPS_MLP_Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super(GPS_MLP_Actor, self).__init__()
        self.output_dim = output_dim
        self.pre_linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GPSConv(channels=hidden_dim, conv=GCNConv(hidden_dim, hidden_dim), heads=heads, dropout=dropout)
        self.conv2 = GPSConv(channels=hidden_dim, conv=GCNConv(hidden_dim, hidden_dim), heads=heads, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        

    def forward(self, x, edge_index):
        x = F.relu(self.pre_linear(x))
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        delta = self.mlp(h)
        delta = (delta + delta.T) / 2
        return delta