# models/gnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    """
    图神经网络编码器 - 解决VNF嵌入中的核心挑战
    
    功能：
    1. 处理可变大小的网络图（节点+边特征）
    2. 编码边缘信息（带宽、延迟、抖动、丢包）
    3. 输出固定大小的状态表示
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(GNNEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 输入特征预处理
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # 图卷积层 - 使用GAT来处理边特征
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # 第一层：处理原始特征
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                           edge_dim=hidden_dim, dropout=0.1)
                )
            else:
                # 后续层：特征传播和聚合
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                           edge_dim=hidden_dim, dropout=0.1)
                )
        
        # 全局池化 - 关键：将可变大小图转换为固定大小
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        # 输出投影层
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Set2Set输出是2*hidden_dim
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data对象或Batch
                - data.x: 节点特征 [num_nodes, node_dim]
                - data.edge_index: 边索引 [2, num_edges] 
                - data.edge_attr: 边特征 [num_edges, edge_dim]
                - data.batch: 批次信息（如果是batch）
        
        Returns:
            graph_embedding: 固定大小的图表示 [batch_size, output_dim]
        """
        
        # 处理单个图和批量图
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()  # 节点特征
        edge_index = data.edge_index  # 边索引
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None  # 边特征
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 特征预处理和归一化
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
            # 边特征归一化 - 重要：确保训练稳定性
            edge_attr = F.normalize(edge_attr, p=2, dim=1)
        
        # 多层图卷积 + 残差连接
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            
            # 图卷积
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化和激活
            if batch is not None:
                # 处理批量数据的归一化
                x = self.batch_norms[i](x)
            else:
                # 单图数据
                if x.size(0) > 1:  # 确保有足够的样本进行批归一化
                    x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            # 残差连接（如果维度匹配）
            if x_residual.size() == x.size():
                x = x + x_residual
        
        # 全局池化 - 关键步骤：图→向量
        if batch is not None:
            # 批量处理
            graph_embedding = self.global_pool(x, batch)
        else:
            # 单图处理 - 创建虚拟batch
            batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = self.global_pool(x, batch_single)
        
        # 输出投影
        graph_embedding = self.output_layers(graph_embedding)
        
        # 最终归一化 - 确保输出稳定性
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        return graph_embedding
    
    def encode_network_state(self, graph, node_features, edge_features):
        """
        便捷方法：直接从网络状态生成编码
        
        Args:
            graph: NetworkX图对象
            node_features: 节点特征矩阵
            edge_features: 边特征矩阵
            
        Returns:
            encoded_state: 编码后的网络状态
        """
        # 转换为PyG Data格式
        edge_list = list(graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # 编码
        with torch.no_grad():
            encoded_state = self.forward(data)
        
        return encoded_state.squeeze(0)  # 移除batch维度
    
    def get_edge_importance(self, data):
        """
        获取边的重要性权重 - 用于分析哪些网络链路最重要
        
        Returns:
            edge_weights: 每条边的重要性分数
        """
        # 这里可以添加注意力权重的提取逻辑
        # 暂时返回平均值
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            return torch.ones(data.edge_attr.size(0))
        else:
            return torch.ones(data.edge_index.size(1))


class EdgeAwareGNNEncoder(GNNEncoder):
    """
    边缘感知的GNN编码器 - 专门为VNF嵌入优化
    
    特殊功能：
    1. 强调边缘特征（带宽、延迟等）的重要性
    2. 添加VNF需求感知机制
    3. 支持多尺度网络表示
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(EdgeAwareGNNEncoder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers)
        
        # VNF需求编码器
        self.vnf_requirement_encoder = nn.Linear(4, hidden_dim)  # CPU, Memory, Bandwidth, Priority
        
        # 边缘重要性评估器
        self.edge_importance_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward_with_vnf_context(self, data, vnf_requirements=None):
        """
        带VNF上下文的前向传播
        
        Args:
            data: 网络图数据
            vnf_requirements: VNF需求 [batch_size, 4] 或 [4]
            
        Returns:
            context_aware_embedding: 上下文感知的图嵌入
        """
        # 标准图编码
        graph_embedding = self.forward(data)
        
        # 如果有VNF需求，融合上下文信息
        if vnf_requirements is not None:
            vnf_requirements = torch.tensor(vnf_requirements, dtype=torch.float32)
            if vnf_requirements.dim() == 1:
                vnf_requirements = vnf_requirements.unsqueeze(0)
            
            vnf_context = self.vnf_requirement_encoder(vnf_requirements)
            
            # 上下文融合
            graph_embedding = graph_embedding + 0.3 * vnf_context
        
        return graph_embedding


# 测试和验证函数
def test_gnn_encoder():
    """测试GNN编码器的功能"""
    print("🧪 测试GNN编码器...")
    
    # 创建测试数据
    num_nodes = 10
    num_edges = 20
    node_dim = 8
    edge_dim = 4
    
    # 模拟网络数据
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 创建编码器
    encoder = GNNEncoder(node_dim=node_dim, edge_dim=edge_dim, output_dim=256)
    
    # 测试编码
    with torch.no_grad():
        output = encoder(data)
    
    print(f"✅ 输入图: {num_nodes}节点, {num_edges}边")
    print(f"✅ 输出向量: {output.shape}")
    print(f"✅ 固定大小编码成功!")
    
    # 测试不同大小的图
    data2 = Data(x=torch.randn(15, node_dim), 
                 edge_index=torch.randint(0, 15, (2, 30)),
                 edge_attr=torch.randn(30, edge_dim))
    
    with torch.no_grad():
        output2 = encoder(data2)
    
    print(f"✅ 不同大小图测试: {output2.shape}")
    print(f"✅ 输出维度一致: {output.shape == output2.shape}")

if __name__ == "__main__":
    test_gnn_encoder()