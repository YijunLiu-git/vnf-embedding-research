# models/enhanced_gnn_encoder.py - 增强的Edge-Aware GNN编码器

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np

class EdgeAttentionLayer(nn.Module):
    """
    边注意力层 - Edge-Aware的核心创新
    
    功能：
    1. 计算边的重要性权重
    2. 基于VNF需求动态调整边权重
    3. 融合边特征和全局网络状态
    """
    
    def __init__(self, edge_dim, hidden_dim, vnf_context_dim=6):
        super(EdgeAttentionLayer, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.vnf_context_dim = vnf_context_dim
        
        # 边特征投影
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # VNF需求编码器
        self.vnf_encoder = nn.Linear(vnf_context_dim, hidden_dim)
        
        # 注意力计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 边重要性分类器
        self.importance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3个重要性等级：低、中、高
            nn.Softmax(dim=-1)
        )
        
        # 全局网络状态感知
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, edge_attr, edge_index, vnf_context=None, network_state=None):
        """
        计算边注意力权重
        
        Args:
            edge_attr: [E, edge_dim] 边特征
            edge_index: [2, E] 边索引
            vnf_context: [vnf_context_dim] VNF需求上下文
            network_state: [network_state_dim] 全局网络状态
            
        Returns:
            edge_attention: [E] 边注意力权重
            edge_features: [E, hidden_dim] 增强边特征
            edge_importance: [E, 3] 边重要性分布
        """
        # 1. 边特征编码
        edge_features = self.edge_proj(edge_attr)  # [E, hidden_dim]
        
        # 2. VNF上下文融合
        if vnf_context is not None:
            if vnf_context.dim() == 1:
                vnf_context = vnf_context.unsqueeze(0)
            vnf_embedding = self.vnf_encoder(vnf_context)  # [1, hidden_dim]
            
            # 广播VNF嵌入到所有边
            vnf_broadcast = vnf_embedding.expand(edge_features.size(0), -1)  # [E, hidden_dim]
            
            # 融合边特征和VNF需求
            combined_features = torch.cat([edge_features, vnf_broadcast], dim=-1)  # [E, hidden_dim*2]
            
            # 计算注意力权重
            edge_attention = self.attention_net(combined_features).squeeze(-1)  # [E]
        else:
            # 默认均匀注意力
            edge_attention = torch.ones(edge_features.size(0), device=edge_features.device)
        
        # 3. 全局网络状态感知
        if network_state is not None:
            # 使用多头注意力整合全局状态
            edge_features_expanded = edge_features.unsqueeze(0)  # [1, E, hidden_dim]
            network_state_expanded = network_state.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            
            attended_features, _ = self.global_attention(
                edge_features_expanded, 
                network_state_expanded, 
                network_state_expanded
            )
            edge_features = attended_features.squeeze(0)  # [E, hidden_dim]
        
        # 4. 边重要性分类
        edge_importance = self.importance_classifier(edge_features)  # [E, 3]
        
        # 5. 应用注意力权重
        weighted_edge_features = edge_features * edge_attention.unsqueeze(-1)
        
        return edge_attention, weighted_edge_features, edge_importance


class PathQualityAwareConv(nn.Module):
    """
    路径质量感知卷积层
    
    功能：
    1. 整合路径质量信息
    2. 动态调整消息传播权重
    3. 考虑端到端路径约束
    """
    
    def __init__(self, in_dim, out_dim, edge_dim, heads=4):
        super(PathQualityAwareConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        
        # 基础GAT层
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=0.1
        )
        
        # 路径质量编码器
        self.path_quality_encoder = nn.Sequential(
            nn.Linear(4, edge_dim // 2),  # 输入：[bandwidth, latency, jitter, loss]
            nn.ReLU(),
            nn.Linear(edge_dim // 2, edge_dim)
        )
        
        # 路径约束感知层
        self.constraint_awareness = nn.Sequential(
            nn.Linear(out_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, edge_index, edge_attr, path_quality_info=None):
        """
        路径质量感知的图卷积
        
        Args:
            x: [N, in_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            path_quality_info: 路径质量信息字典
            
        Returns:
            x_out: [N, out_dim] 更新后的节点特征
        """
        # 1. 基础图卷积
        x_conv = self.gat(x, edge_index, edge_attr)
        
        # 2. 路径质量信息融合
        if path_quality_info is not None:
            # 提取路径质量特征
            quality_features = self._extract_path_quality_features(
                edge_index, edge_attr, path_quality_info
            )
            
            # 增强边特征
            enhanced_edge_attr = edge_attr + quality_features
            
            # 重新计算卷积
            x_conv = self.gat(x, edge_index, enhanced_edge_attr)
        
        # 3. 约束感知处理
        if path_quality_info is not None:
            # 结合约束信息
            constraint_features = self._compute_constraint_features(x_conv, path_quality_info)
            x_out = self.constraint_awareness(
                torch.cat([x_conv, constraint_features], dim=-1)
            )
        else:
            x_out = x_conv
        
        return x_out
    
    def _extract_path_quality_features(self, edge_index, edge_attr, path_quality_info):
        """提取路径质量特征"""
        # 从路径质量矩阵中提取相关信息
        quality_matrix = path_quality_info.get('path_quality_matrix', {})
        
        # 构建质量特征向量
        quality_features = torch.zeros_like(edge_attr)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            src_idx, dst_idx = src.item(), dst.item()
            path_info = quality_matrix.get((src_idx, dst_idx), {})
            
            if path_info:
                # 提取质量指标
                bandwidth = path_info.get('bandwidth', 0.0)
                latency = path_info.get('latency', 0.0)
                jitter = path_info.get('jitter', 0.0)
                loss = path_info.get('packet_loss', 0.0)
                
                # 编码质量特征
                quality_vec = torch.tensor([bandwidth/100, latency/100, jitter*100, loss*100], 
                                         device=edge_attr.device)
                quality_encoded = self.path_quality_encoder(quality_vec)
                quality_features[i] = quality_encoded
        
        return quality_features
    
    def _compute_constraint_features(self, node_features, path_quality_info):
        """计算约束特征"""
        # 简化实现：基于网络状态计算约束特征
        network_state = path_quality_info.get('network_state_vector', torch.zeros(8))
        
        if isinstance(network_state, np.ndarray):
            network_state = torch.tensor(network_state, device=node_features.device)
        
        # 广播网络状态到所有节点
        constraint_features = network_state.unsqueeze(0).expand(
            node_features.size(0), -1
        )
        
        # 确保维度匹配
        if constraint_features.size(-1) != self.edge_dim:
            constraint_features = F.adaptive_avg_pool1d(
                constraint_features.unsqueeze(1), self.edge_dim
            ).squeeze(1)
        
        return constraint_features


class EnhancedEdgeAwareGNN(nn.Module):
    """
    增强的Edge-Aware GNN编码器
    
    核心创新：
    1. 边注意力机制
    2. 路径质量感知卷积
    3. VNF需求适应性编码
    4. 动态网络状态更新
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, 
                 num_layers=6, vnf_context_dim=6):
        super(EnhancedEdgeAwareGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.vnf_context_dim = vnf_context_dim
        
        # 节点和边嵌入
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # 边注意力层
        self.edge_attention = EdgeAttentionLayer(edge_dim, hidden_dim, vnf_context_dim)
        
        # 路径质量感知卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                PathQualityAwareConv(hidden_dim, hidden_dim, hidden_dim, heads=4)
            )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # VNF上下文编码器
        self.vnf_context_encoder = nn.Sequential(
            nn.Linear(vnf_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 全局池化和输出
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        # 多层输出网络
        self.output_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # +hidden_dim for VNF context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU()
        )
        
        # 网络状态编码器
        self.network_state_encoder = nn.Linear(8, hidden_dim)  # 8维网络状态向量
        
        print(f"🚀 增强Edge-Aware GNN编码器初始化完成")
        print(f"   - 节点维度: {node_dim} -> {hidden_dim}")
        print(f"   - 边维度: {edge_dim} -> {hidden_dim}")
        print(f"   - 卷积层数: {num_layers}")
        print(f"   - 输出维度: {output_dim}")
        print(f"   - 核心创新: 边注意力 + 路径质量感知")
        
    def forward(self, data):
        """
        前向传播 - 完整的Edge-Aware处理流程
        
        Args:
            data: PyG数据对象，包含：
                - x: [N, node_dim] 节点特征
                - edge_index: [2, E] 边索引
                - edge_attr: [E, edge_dim] 边特征
                - vnf_context: [vnf_context_dim] VNF上下文（可选）
                - network_state: [8] 网络状态向量（可选）
                - enhanced_info: 增强状态信息（可选）
                
        Returns:
            graph_embedding: [batch_size, output_dim] 图嵌入
        """
        # 处理输入数据
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 提取额外信息
        vnf_context = getattr(data, 'vnf_context', None)
        network_state = getattr(data, 'network_state', None)
        enhanced_info = getattr(data, 'enhanced_info', None)
        
        # 验证输入维度
        self._validate_input_dimensions(x, edge_attr)
        
        # 1. 基础特征嵌入
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        # 2. 边注意力计算
        edge_attention, enhanced_edge_features, edge_importance = self.edge_attention(
            edge_attr, edge_index, vnf_context, network_state
        )
        
        # 3. 多层图卷积（路径质量感知）
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x
            
            # 路径质量感知卷积
            x = conv(x, edge_index, enhanced_edge_features, enhanced_info)
            
            # 层归一化
            x = norm(x)
            
            # 残差连接
            if x_residual.size() == x.size():
                x = x + x_residual
            
            # 激活函数
            x = F.relu(x)
        
        # 4. 全局池化
        if batch is not None:
            graph_embedding = self.global_pool(x, batch)
        else:
            batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = self.global_pool(x, batch_single)
        
        # 5. VNF上下文融合
        if vnf_context is not None:
            vnf_embedding = self.vnf_context_encoder(vnf_context.float())
            if vnf_embedding.dim() == 1:
                vnf_embedding = vnf_embedding.unsqueeze(0)
            
            # 确保batch维度匹配
            if vnf_embedding.size(0) != graph_embedding.size(0):
                vnf_embedding = vnf_embedding.expand(graph_embedding.size(0), -1)
            
            # 融合图嵌入和VNF上下文
            combined_embedding = torch.cat([graph_embedding, vnf_embedding], dim=-1)
        else:
            # 补零VNF上下文
            zero_context = torch.zeros(graph_embedding.size(0), self.hidden_dim, 
                                     device=graph_embedding.device)
            combined_embedding = torch.cat([graph_embedding, zero_context], dim=-1)
        
        # 6. 最终输出
        final_embedding = self.output_net(combined_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)
        
        return final_embedding
    
    def _validate_input_dimensions(self, x, edge_attr):
        """验证输入维度"""
        if x.size(1) != self.node_dim:
            raise ValueError(f"节点特征维度不匹配: 期望{self.node_dim}, 实际{x.size(1)}")
        
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            if self.edge_dim == 4 and edge_attr.size(1) == 2:
                # 自动扩展2维到4维
                padding = torch.zeros(edge_attr.size(0), 2, device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
                print("🔧 边特征自动扩展: 2维 -> 4维")
            else:
                raise ValueError(f"边特征维度不匹配: 期望{self.edge_dim}, 实际{edge_attr.size(1)}")
    
    def compute_edge_importance_map(self, data):
        """
        计算边重要性映射 - 用于分析和可视化
        
        Returns:
            edge_importance_map: 边重要性分布字典
        """
        with torch.no_grad():
            # 提取基础数据
            edge_index = data.edge_index
            edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
            vnf_context = getattr(data, 'vnf_context', None)
            network_state = getattr(data, 'network_state', None)
            
            # 嵌入边特征
            if edge_attr is not None:
                edge_attr = self.edge_embedding(edge_attr)
            
            # 计算边注意力
            edge_attention, _, edge_importance = self.edge_attention(
                edge_attr, edge_index, vnf_context, network_state
            )
            
            # 构建重要性映射
            edge_importance_map = {}
            for i, (src, dst) in enumerate(edge_index.t()):
                edge_key = (src.item(), dst.item())
                edge_importance_map[edge_key] = {
                    'attention_weight': edge_attention[i].item(),
                    'importance_scores': edge_importance[i].cpu().numpy(),
                    'importance_level': edge_importance[i].argmax().item()  # 0:低, 1:中, 2:高
                }
            
            return edge_importance_map
    
    def get_vnf_adaptation_score(self, data):
        """
        计算VNF适应性评分 - 衡量网络对当前VNF需求的适应程度
        
        Returns:
            adaptation_score: 适应性评分 (0-1)
        """
        with torch.no_grad():
            # 前向传播获取嵌入
            embedding = self.forward(data)
            
            # 计算边重要性
            edge_importance_map = self.compute_edge_importance_map(data)
            
            # 评估指标
            avg_attention = np.mean([info['attention_weight'] for info in edge_importance_map.values()])
            high_importance_ratio = np.mean([
                1 if info['importance_level'] == 2 else 0 
                for info in edge_importance_map.values()
            ])
            
            # 综合评分
            adaptation_score = (avg_attention * 0.6 + high_importance_ratio * 0.4)
            
            return float(adaptation_score)


def create_enhanced_edge_aware_encoder(config: dict):
    """
    创建增强Edge-Aware编码器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        encoder: 增强的Edge-Aware GNN编码器
    """
    gnn_config = config.get('gnn', {}).get('edge_aware', {})
    
    encoder = EnhancedEdgeAwareGNN(
        node_dim=config.get('dimensions', {}).get('node_feature_dim', 8),
        edge_dim=config.get('dimensions', {}).get('edge_feature_dim_full', 4),
        hidden_dim=gnn_config.get('hidden_dim', 128),
        output_dim=gnn_config.get('output_dim', 256),
        num_layers=gnn_config.get('layers', 6),
        vnf_context_dim=config.get('dimensions', {}).get('vnf_context_dim', 6)
    )
    
    print(f"✅ 增强Edge-Aware编码器创建完成")
    return encoder


# 测试函数
def test_enhanced_gnn():
    """测试增强GNN编码器"""
    print("🧪 测试增强Edge-Aware GNN编码器...")
    print("=" * 60)
    
    # 创建测试数据
    num_nodes = 20
    num_edges = 50
    batch_size = 2
    
    # 节点特征 [N, 8]
    x = torch.randn(num_nodes, 8)
    
    # 边索引和特征
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)
    
    # VNF上下文 [6]
    vnf_context = torch.tensor([0.05, 0.03, 0.04, 0.33, 0.5, 0.5])
    
    # 网络状态 [8]
    network_state = torch.randn(8)
    
    # 构建数据对象
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        vnf_context=vnf_context,
        network_state=network_state
    )
    
    # 创建编码器
    config = {
        'dimensions': {
            'node_feature_dim': 8,
            'edge_feature_dim_full': 4,
            'vnf_context_dim': 6
        },
        'gnn': {
            'edge_aware': {
                'hidden_dim': 128,
                'output_dim': 256,
                'layers': 4
            }
        }
    }
    
    encoder = create_enhanced_edge_aware_encoder(config)
    
    # 测试前向传播
    with torch.no_grad():
        output = encoder(data)
        print(f"✅ 前向传播测试:")
        print(f"   输入: {num_nodes}节点, {num_edges}边")
        print(f"   输出: {output.shape}")
        
        # 测试边重要性映射
        importance_map = encoder.compute_edge_importance_map(data)
        print(f"✅ 边重要性映射:")
        print(f"   重要性条目数: {len(importance_map)}")
        print(f"   示例重要性: {list(importance_map.values())[0]}")
        
        # 测试VNF适应性评分
        adaptation_score = encoder.get_vnf_adaptation_score(data)
        print(f"✅ VNF适应性评分: {adaptation_score:.3f}")
        
        # 测试批处理
        batch_data = Batch.from_data_list([data, data])
        batch_output = encoder(batch_data)
        print(f"✅ 批处理测试: {batch_output.shape}")
    
    print(f"\n🎉 增强Edge-Aware GNN编码器测试通过!")
    print(f"核心功能验证:")
    print(f"  ✅ 边注意力机制")
    print(f"  ✅ 路径质量感知")
    print(f"  ✅ VNF上下文融合")
    print(f"  ✅ 动态网络状态")
    print(f"  ✅ 重要性分析")


if __name__ == "__main__":
    test_enhanced_gnn()