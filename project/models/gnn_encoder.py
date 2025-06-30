import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()

        # 改进后的边权重生成器：两层 MLP + ReLU，输出 scalar 权重
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),
            nn.ReLU()  # 保证非负权重，避免 GCN NaN
        )

        # 两层 GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # LayerNorm 稳定训练
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Dropout 提升泛化能力
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        edge_weight = self.edge_embedding(data.edge_attr).squeeze()  # shape: [num_edges]

        x = self.conv1(data.x, data.edge_index, edge_weight=edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, edge_weight=edge_weight)
        x = self.norm2(x)
        return x