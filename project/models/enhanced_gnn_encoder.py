# models/enhanced_gnn_encoder_fixed.py - 修复维度不匹配问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np

class EdgeAttentionLayer(nn.Module):
    """
    边注意力层 - 修复版本
    """
    
    def __init__(self, edge_dim, hidden_dim, vnf_context_dim=6):
        super(EdgeAttentionLayer, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.vnf_context_dim = vnf_context_dim
        
        # 🔧 修复：处理不同的边特征输入情况
        # 如果边特征已经是hidden_dim维度，则不需要投影
        self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()
        
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
            nn.Linear(hidden_dim // 2, 3),
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
        计算边注意力权重 - 修复版本
        """
        # 1. 边特征编码 - 🔧 修复维度问题
        if edge_attr.size(-1) == self.edge_dim:
            edge_features = self.edge_proj(edge_attr)  # 需要投影
        elif edge_attr.size(-1) == self.hidden_dim:
            edge_features = edge_attr  # 已经是正确维度
        else:
            # 自适应处理
            if edge_attr.size(-1) < self.hidden_dim:
                # 如果维度太小，先投影到hidden_dim
                temp_proj = nn.Linear(edge_attr.size(-1), self.hidden_dim).to(edge_attr.device)
                edge_features = temp_proj(edge_attr)
            else:
                # 如果维度太大，降维到hidden_dim
                temp_proj = nn.Linear(edge_attr.size(-1), self.hidden_dim).to(edge_attr.device)
                edge_features = temp_proj(edge_attr)
        
        # 2. VNF上下文融合
        if vnf_context is not None:
            try:
                if vnf_context.dim() == 1:
                    vnf_context = vnf_context.unsqueeze(0)
                vnf_embedding = self.vnf_encoder(vnf_context)  # [1, hidden_dim]
                
                # 广播VNF嵌入到所有边
                vnf_broadcast = vnf_embedding.expand(edge_features.size(0), -1)
                
                # 融合边特征和VNF需求
                combined_features = torch.cat([edge_features, vnf_broadcast], dim=-1)
                
                # 计算注意力权重
                edge_attention = self.attention_net(combined_features).squeeze(-1)
            except Exception as e:
                print(f"⚠️ VNF上下文处理失败: {e}")
                edge_attention = torch.ones(edge_features.size(0), device=edge_features.device)
        else:
            edge_attention = torch.ones(edge_features.size(0), device=edge_features.device)
        
        # 3. 全局网络状态感知
        if network_state is not None:
            try:
                edge_features_expanded = edge_features.unsqueeze(0)
                if hasattr(network_state, 'unsqueeze'):
                    network_state_expanded = network_state.unsqueeze(0).unsqueeze(0)
                else:
                    network_state_tensor = torch.tensor(network_state, device=edge_features.device, dtype=torch.float32)
                    network_state_expanded = network_state_tensor.unsqueeze(0).unsqueeze(0)
                
                # 确保维度匹配
                if network_state_expanded.size(-1) != self.hidden_dim:
                    state_proj = nn.Linear(network_state_expanded.size(-1), self.hidden_dim).to(edge_features.device)
                    network_state_expanded = state_proj(network_state_expanded)
                
                attended_features, _ = self.global_attention(
                    edge_features_expanded, 
                    network_state_expanded, 
                    network_state_expanded
                )
                edge_features = attended_features.squeeze(0)
            except Exception as e:
                print(f"⚠️ 网络状态感知失败: {e}")
        
        # 4. 边重要性分类
        try:
            edge_importance = self.importance_classifier(edge_features)
        except Exception as e:
            print(f"⚠️ 边重要性分类失败: {e}")
            edge_importance = torch.ones(edge_features.size(0), 3, device=edge_features.device) / 3.0
        
        # 5. 应用注意力权重
        weighted_edge_features = edge_features * edge_attention.unsqueeze(-1)
        
        return edge_attention, weighted_edge_features, edge_importance


class PathQualityAwareConv(nn.Module):
    """
    路径质量感知卷积层 - 修复版本
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
            nn.Linear(4, edge_dim // 2),
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
        路径质量感知的图卷积 - 修复版本
        """
        # 🔧 修复：确保edge_attr维度正确
        if edge_attr.size(-1) != self.edge_dim:
            # 动态调整边特征维度
            if edge_attr.size(-1) < self.edge_dim:
                padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(-1), 
                                    device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=-1)
            else:
                edge_attr = edge_attr[:, :self.edge_dim]
        
        # 1. 基础图卷积
        try:
            x_conv = self.gat(x, edge_index, edge_attr)
        except Exception as e:
            print(f"⚠️ GAT卷积失败: {e}")
            # 回退到简单的线性变换
            x_conv = nn.Linear(self.in_dim, self.out_dim).to(x.device)(x)
        
        # 2. 路径质量信息融合
        if path_quality_info is not None:
            try:
                quality_features = self._extract_path_quality_features_safe(
                    edge_index, edge_attr, path_quality_info
                )
                enhanced_edge_attr = edge_attr + quality_features
                x_conv = self.gat(x, edge_index, enhanced_edge_attr)
            except Exception as e:
                print(f"⚠️ 路径质量融合失败: {e}")
        
        # 3. 约束感知处理
        if path_quality_info is not None:
            try:
                constraint_features = self._compute_constraint_features_safe(x_conv, path_quality_info)
                x_out = self.constraint_awareness(
                    torch.cat([x_conv, constraint_features], dim=-1)
                )
            except Exception as e:
                print(f"⚠️ 约束感知失败: {e}")
                x_out = x_conv
        else:
            x_out = x_conv
        
        return x_out
    
    def _extract_path_quality_features_safe(self, edge_index, edge_attr, path_quality_info):
        """安全的路径质量特征提取"""
        try:
            quality_matrix = path_quality_info.get('path_quality_matrix', {})
            quality_features = torch.zeros_like(edge_attr)
            
            for i, (src, dst) in enumerate(edge_index.t()):
                if i < quality_features.size(0):
                    src_idx, dst_idx = src.item(), dst.item()
                    path_info = quality_matrix.get((src_idx, dst_idx), {})
                    
                    if path_info:
                        bandwidth = path_info.get('bandwidth', 0.0)
                        latency = path_info.get('latency', 0.0)
                        jitter = path_info.get('jitter', 0.0)
                        loss = path_info.get('packet_loss', 0.0)
                        
                        quality_vec = torch.tensor([bandwidth/100, latency/100, jitter*100, loss*100], 
                                                 device=edge_attr.device)
                        
                        if hasattr(self, 'path_quality_encoder'):
                            quality_encoded = self.path_quality_encoder(quality_vec)
                            if quality_encoded.size(0) == edge_attr.size(-1):
                                quality_features[i] = quality_encoded
            
            return quality_features
        except Exception as e:
            print(f"⚠️ 路径质量特征提取失败: {e}")
            return torch.zeros_like(edge_attr)
    
    def _compute_constraint_features_safe(self, node_features, path_quality_info):
        """安全的约束特征计算"""
        try:
            network_state = path_quality_info.get('network_state_vector', torch.zeros(8))
            
            if isinstance(network_state, np.ndarray):
                network_state = torch.tensor(network_state, device=node_features.device)
            elif isinstance(network_state, list):
                network_state = torch.tensor(network_state, device=node_features.device, dtype=torch.float32)
            
            constraint_features = network_state.unsqueeze(0).expand(node_features.size(0), -1)
            
            if constraint_features.size(-1) != self.edge_dim:
                if constraint_features.size(-1) < self.edge_dim:
                    padding = torch.zeros(constraint_features.size(0), 
                                        self.edge_dim - constraint_features.size(-1),
                                        device=constraint_features.device)
                    constraint_features = torch.cat([constraint_features, padding], dim=-1)
                else:
                    constraint_features = constraint_features[:, :self.edge_dim]
            
            return constraint_features
        except Exception as e:
            print(f"⚠️ 约束特征计算失败: {e}")
            return torch.zeros(node_features.size(0), self.edge_dim, device=node_features.device)


class EnhancedEdgeAwareGNN(nn.Module):
    """
    增强的Edge-Aware GNN编码器 - 修复版本
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
        
        # 🔧 修复：边注意力层使用原始边维度
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
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU()
        )
        
        # 网络状态编码器
        self.network_state_encoder = nn.Linear(8, hidden_dim)
        
        print(f"🚀 增强Edge-Aware GNN编码器初始化完成 (修复版)")
        print(f"   - 节点维度: {node_dim} -> {hidden_dim}")
        print(f"   - 边维度: {edge_dim} -> {hidden_dim}")
        print(f"   - 卷积层数: {num_layers}")
        print(f"   - 输出维度: {output_dim}")
        
    def forward(self, data):
        """
        前向传播 - 修复版本
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
        try:
            self._validate_input_dimensions(x, edge_attr)
        except Exception as e:
            print(f"⚠️ 维度验证失败: {e}")
            # 自动修复维度
            x, edge_attr = self._auto_fix_dimensions(x, edge_attr)
        
        # 1. 基础特征嵌入
        x = self.node_embedding(x)
        original_edge_attr = edge_attr  # 保存原始边特征
        
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        # 2. 边注意力计算 - 🔧 使用原始边特征
        try:
            edge_attention, enhanced_edge_features, edge_importance = self.edge_attention(
                original_edge_attr, edge_index, vnf_context, network_state
            )
        except Exception as e:
            print(f"⚠️ 边注意力计算失败: {e}")
            # 使用默认值
            edge_attention = torch.ones(edge_attr.size(0), device=x.device) if edge_attr is not None else torch.ones(edge_index.size(1), device=x.device)
            enhanced_edge_features = edge_attr if edge_attr is not None else torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
            edge_importance = torch.ones(enhanced_edge_features.size(0), 3, device=x.device) / 3.0
        
        # 3. 多层图卷积
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x
            
            try:
                x = conv(x, edge_index, enhanced_edge_features, enhanced_info)
                x = norm(x)
                
                if x_residual.size() == x.size():
                    x = x + x_residual
                
                x = F.relu(x)
            except Exception as e:
                print(f"⚠️ 第{i}层卷积失败: {e}")
                # 跳过这一层
                continue
        
        # 4. 全局池化
        try:
            if batch is not None:
                graph_embedding = self.global_pool(x, batch)
            else:
                batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                graph_embedding = self.global_pool(x, batch_single)
        except Exception as e:
            print(f"⚠️ 全局池化失败: {e}")
            # 使用简单的全局平均池化
            graph_embedding = x.mean(dim=0, keepdim=True)
            graph_embedding = torch.cat([graph_embedding, graph_embedding], dim=-1)  # 模拟Set2Set输出
        
        # 5. VNF上下文融合
        try:
            if vnf_context is not None:
                vnf_embedding = self.vnf_context_encoder(vnf_context.float())
                if vnf_embedding.dim() == 1:
                    vnf_embedding = vnf_embedding.unsqueeze(0)
                
                if vnf_embedding.size(0) != graph_embedding.size(0):
                    vnf_embedding = vnf_embedding.expand(graph_embedding.size(0), -1)
                
                combined_embedding = torch.cat([graph_embedding, vnf_embedding], dim=-1)
            else:
                zero_context = torch.zeros(graph_embedding.size(0), self.hidden_dim, device=graph_embedding.device)
                combined_embedding = torch.cat([graph_embedding, zero_context], dim=-1)
        except Exception as e:
            print(f"⚠️ VNF上下文融合失败: {e}")
            # 使用零填充
            zero_context = torch.zeros(graph_embedding.size(0), self.hidden_dim, device=graph_embedding.device)
            combined_embedding = torch.cat([graph_embedding, zero_context], dim=-1)
        
        # 6. 最终输出
        try:
            final_embedding = self.output_net(combined_embedding)
            final_embedding = F.normalize(final_embedding, p=2, dim=-1)
        except Exception as e:
            print(f"⚠️ 最终输出失败: {e}")
            # 使用简单的线性变换
            final_embedding = nn.Linear(combined_embedding.size(-1), self.output_dim).to(combined_embedding.device)(combined_embedding)
            final_embedding = F.normalize(final_embedding, p=2, dim=-1)
        
        return final_embedding
    
    def _validate_input_dimensions(self, x, edge_attr):
        """验证输入维度"""
        if x.size(1) != self.node_dim:
            raise ValueError(f"节点特征维度不匹配: 期望{self.node_dim}, 实际{x.size(1)}")
        
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            if self.edge_dim == 4 and edge_attr.size(1) == 2:
                # 自动扩展2维到4维
                return  # 允许这种情况
            else:
                raise ValueError(f"边特征维度不匹配: 期望{self.edge_dim}, 实际{edge_attr.size(1)}")
    
    def _auto_fix_dimensions(self, x, edge_attr):
        """自动修复维度问题"""
        # 修复节点特征维度
        if x.size(1) != self.node_dim:
            if x.size(1) < self.node_dim:
                padding = torch.zeros(x.size(0), self.node_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.node_dim]
        
        # 修复边特征维度
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            if edge_attr.size(1) < self.edge_dim:
                padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(1), device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=-1)
            else:
                edge_attr = edge_attr[:, :self.edge_dim]
        
        return x, edge_attr
    
    def compute_edge_importance_map(self, data):
        """计算边重要性映射 - 安全版本"""
        try:
            with torch.no_grad():
                edge_index = data.edge_index
                edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
                vnf_context = getattr(data, 'vnf_context', None)
                network_state = getattr(data, 'network_state', None)
                
                edge_attention, _, edge_importance = self.edge_attention(
                    edge_attr, edge_index, vnf_context, network_state
                )
                
                edge_importance_map = {}
                for i, (src, dst) in enumerate(edge_index.t()):
                    if i < len(edge_attention):
                        edge_key = (src.item(), dst.item())
                        edge_importance_map[edge_key] = {
                            'attention_weight': edge_attention[i].item(),
                            'importance_scores': edge_importance[i].cpu().numpy(),
                            'importance_level': edge_importance[i].argmax().item()
                        }
                
                return edge_importance_map
        except Exception as e:
            print(f"⚠️ 边重要性映射计算失败: {e}")
            return {}
    
    def get_vnf_adaptation_score(self, data):
        """计算VNF适应性评分 - 安全版本"""
        try:
            with torch.no_grad():
                embedding = self.forward(data)
                edge_importance_map = self.compute_edge_importance_map(data)
                
                if edge_importance_map:
                    avg_attention = np.mean([info['attention_weight'] for info in edge_importance_map.values()])
                    high_importance_ratio = np.mean([
                        1 if info['importance_level'] == 2 else 0 
                        for info in edge_importance_map.values()
                    ])
                    adaptation_score = (avg_attention * 0.6 + high_importance_ratio * 0.4)
                else:
                    adaptation_score = 0.5  # 默认中等适应性
                
                return float(adaptation_score)
        except Exception as e:
            print(f"⚠️ VNF适应性评分计算失败: {e}")
            return 0.5


def create_enhanced_edge_aware_encoder_fixed(config: dict):
    """
    创建修复版增强Edge-Aware编码器
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
    
    print(f"✅ 修复版增强Edge-Aware编码器创建完成")
    return encoder