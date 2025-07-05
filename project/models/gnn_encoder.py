# models/gnn_encoder.py - 修复Baseline维度不匹配问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    """
    修复版GNN编码器 - 专门解决Baseline边特征维度不匹配问题
    
    🔧 核心修复：
    1. 自动处理4维→2维边特征降维
    2. 确保8维节点特征输入
    3. 保证256维输出
    """
    
    def __init__(self, node_dim=8, edge_dim=2, hidden_dim=128, output_dim=256, num_layers=3):
        super(GNNEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim  # 期望的边特征维度（通常是2）
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # 🔧 关键修复：边特征自适应嵌入层
        # 无论输入多少维，都能正确处理
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # GAT卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                       edge_dim=hidden_dim, dropout=0.1)
            )
        
        # 全局池化
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        # 🔧 修复：确保输出维度正确
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Set2Set输出2*hidden_dim
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),       # 确保256维输出
            nn.ReLU()
        )
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        print(f"✅ GNN编码器初始化: 节点{node_dim}维 -> 边{edge_dim}维 -> 输出{output_dim}维")
        
    def forward(self, data):
        """
        前向传播 - 自动处理边特征维度
        """
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 🔧 关键修复1：节点特征维度验证
        if x.size(1) != self.node_dim:
            raise ValueError(f"❌ 节点特征维度不匹配: 期望{self.node_dim}维，实际{x.size(1)}维")
        
        # 🔧 关键修复2：边特征自动降维处理
        if edge_attr is not None:
            actual_edge_dim = edge_attr.size(1)
            
            if actual_edge_dim != self.edge_dim:
                print(f"🔧 边特征维度自动调整: {actual_edge_dim}维 -> {self.edge_dim}维")
                
                if actual_edge_dim > self.edge_dim:
                    # 降维：截取前N维
                    edge_attr = edge_attr[:, :self.edge_dim]
                    print(f"   截取前{self.edge_dim}维: [带宽, 延迟]")
                    
                elif actual_edge_dim < self.edge_dim:
                    # 升维：用零填充
                    padding_dims = self.edge_dim - actual_edge_dim
                    padding = torch.zeros(edge_attr.size(0), padding_dims, device=edge_attr.device)
                    edge_attr = torch.cat([edge_attr, padding], dim=1)
                    print(f"   零填充到{self.edge_dim}维")
        
        # 特征嵌入
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
            edge_attr = F.normalize(edge_attr, p=2, dim=1)
        
        # GNN卷积
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            if batch is not None:
                x = self.batch_norms[i](x)
            else:
                if x.size(0) > 1:
                    x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            # 残差连接
            if x_residual.size() == x.size():
                x = x + x_residual
        
        # 全局池化
        if batch is not None:
            graph_embedding = self.global_pool(x, batch)
        else:
            batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = self.global_pool(x, batch_single)
        
        # 🔧 验证池化后维度
        expected_pooled_dim = 2 * self.hidden_dim
        if graph_embedding.size(-1) != expected_pooled_dim:
            print(f"⚠️ 池化维度异常: 期望{expected_pooled_dim}, 实际{graph_embedding.size(-1)}")
        
        # 输出层
        graph_embedding = self.output_layers(graph_embedding)
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        # 🔧 最终验证输出维度
        if graph_embedding.size(-1) != self.output_dim:
            print(f"⚠️ 输出维度异常: 期望{self.output_dim}, 实际{graph_embedding.size(-1)}")
        
        return graph_embedding


class EdgeAwareGNNEncoder(GNNEncoder):
    """
    边感知GNN编码器 - 继承基础修复逻辑
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=4):
        # Edge-aware模式使用4维边特征
        super(EdgeAwareGNNEncoder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers)
        
        # VNF需求编码器
        self.vnf_requirement_encoder = nn.Linear(6, hidden_dim)
        
        # 边重要性网络
        self.edge_importance_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim + hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"✅ EdgeAware编码器初始化: 支持{edge_dim}维边特征")
        
    def forward_with_vnf_context(self, data, vnf_context=None):
        """带VNF上下文的前向传播"""
        # 基础图编码
        graph_embedding = self.forward(data)
        
        # VNF上下文融合
        if vnf_context is not None:
            if isinstance(vnf_context, torch.Tensor):
                vnf_tensor = vnf_context.float()
            else:
                vnf_tensor = torch.tensor(vnf_context, dtype=torch.float32)
            
            if vnf_tensor.dim() == 1:
                vnf_tensor = vnf_tensor.unsqueeze(0)
            
            # VNF上下文编码
            vnf_embedding = self.vnf_requirement_encoder(vnf_tensor)
            
            # 特征融合
            if graph_embedding.size(0) == vnf_embedding.size(0):
                fused_features = torch.cat([graph_embedding, vnf_embedding], dim=1)
                enhanced_embedding = self.feature_fusion(fused_features)
            else:
                # 广播处理
                enhanced_embedding = graph_embedding + 0.3 * vnf_embedding.mean(dim=0, keepdim=True)
            
            return enhanced_embedding
        else:
            return graph_embedding


def create_gnn_encoder(config: dict, mode: str = 'edge_aware'):
    """
    创建GNN编码器的工厂函数 - 修复版
    
    🔧 关键修复：自动处理baseline的边特征维度问题
    """
    
    if mode == 'edge_aware':
        gnn_config = config.get('gnn', {}).get('edge_aware', {})
        # Edge-aware使用4维边特征
        encoder = EdgeAwareGNNEncoder(
            node_dim=8,  # 固定8维节点特征
            edge_dim=4,  # 4维边特征：[bandwidth, latency, jitter, packet_loss]
            hidden_dim=gnn_config.get('hidden_dim', 128),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 4)
        )
        print(f"✅ 创建Edge-aware GNN编码器: 4维边特征")
        
    else:  # baseline
        gnn_config = config.get('gnn', {}).get('baseline', {})
        # 🔧 关键修复：Baseline仍然使用2维，但GNN会自动降维
        encoder = GNNEncoder(
            node_dim=8,  # 固定8维节点特征
            edge_dim=2,  # 2维边特征：[bandwidth, latency]（GNN会自动截取）
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 3)
        )
        print(f"✅ 创建Baseline GNN编码器: 2维边特征（自动降维）")
    
    return encoder


# 专门的测试函数
def test_baseline_dimension_fix():
    """测试Baseline维度修复"""
    print("🧪 测试Baseline边特征维度修复...")
    print("=" * 60)
    
    # 模拟真实场景：环境输出4维边特征，Baseline期望2维
    num_nodes = 10
    num_edges = 20
    
    # 模拟环境数据：8维节点特征 + 4维边特征
    x = torch.randn(num_nodes, 8)  # 8维节点特征
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)  # 4维边特征（环境输出）
    
    print(f"📊 测试数据:")
    print(f"   节点特征: {x.shape}")
    print(f"   边特征: {edge_attr.shape}")
    
    # 测试1: Baseline编码器（期望2维边特征）
    print(f"\n🧪 测试1: Baseline编码器处理4维边特征")
    baseline_encoder = GNNEncoder(node_dim=8, edge_dim=2, output_dim=256)
    
    data_baseline = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    try:
        with torch.no_grad():
            output_baseline = baseline_encoder(data_baseline)
        print(f"✅ Baseline测试成功: {output_baseline.shape}")
        assert output_baseline.shape[1] == 256, f"输出维度应为256，实际{output_baseline.shape[1]}"
        print(f"   ✓ 输出维度正确: 256维")
        
    except Exception as e:
        print(f"❌ Baseline测试失败: {e}")
        return False
    
    # 测试2: Edge-aware编码器（期望4维边特征）
    print(f"\n🧪 测试2: Edge-aware编码器处理4维边特征")
    edge_aware_encoder = EdgeAwareGNNEncoder(node_dim=8, edge_dim=4, output_dim=256)
    
    data_edge_aware = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    try:
        with torch.no_grad():
            output_edge_aware = edge_aware_encoder(data_edge_aware)
        print(f"✅ Edge-aware测试成功: {output_edge_aware.shape}")
        assert output_edge_aware.shape[1] == 256, f"输出维度应为256，实际{output_edge_aware.shape[1]}"
        print(f"   ✓ 输出维度正确: 256维")
        
    except Exception as e:
        print(f"❌ Edge-aware测试失败: {e}")
        return False
    
    # 测试3: 维度一致性验证
    print(f"\n🧪 测试3: 输出维度一致性验证")
    assert output_baseline.shape == output_edge_aware.shape, "两种模式输出维度不一致"
    print(f"✅ 输出维度一致性测试通过: {output_baseline.shape}")
    
    # 测试4: 配置文件工厂函数
    print(f"\n🧪 测试4: 配置文件工厂函数")
    test_config = {
        'gnn': {
            'edge_aware': {'hidden_dim': 128, 'output_dim': 256, 'layers': 4, 'edge_dim': 4},
            'baseline': {'hidden_dim': 64, 'output_dim': 256, 'layers': 3, 'edge_dim': 2}
        }
    }
    
    try:
        baseline_encoder_config = create_gnn_encoder(test_config, mode='baseline')
        edge_aware_encoder_config = create_gnn_encoder(test_config, mode='edge_aware')
        
        # 测试配置创建的编码器
        with torch.no_grad():
            output_baseline_config = baseline_encoder_config(data_baseline)
            output_edge_aware_config = edge_aware_encoder_config(data_edge_aware)
            
        print(f"✅ 配置工厂函数测试成功")
        print(f"   Baseline输出: {output_baseline_config.shape}")
        print(f"   Edge-aware输出: {output_edge_aware_config.shape}")
        
    except Exception as e:
        print(f"❌ 配置工厂函数测试失败: {e}")
        return False
    
    print(f"\n🎉 所有Baseline维度修复测试通过!")
    print(f"核心修复点:")
    print(f"  ✅ 自动截取4维→2维边特征")
    print(f"  ✅ 保持8维节点特征输入") 
    print(f"  ✅ 确保256维输出")
    print(f"  ✅ 兼容现有配置文件")
    
    return True


def test_edge_dimension_scenarios():
    """测试各种边特征维度场景"""
    print("\n🧪 测试边特征维度处理场景...")
    
    # 准备测试数据
    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 场景测试
    scenarios = [
        (2, "2维边特征 → 2维期望"),
        (4, "4维边特征 → 2维期望"), 
        (6, "6维边特征 → 2维期望"),
        (1, "1维边特征 → 2维期望")
    ]
    
    baseline_encoder = GNNEncoder(node_dim=8, edge_dim=2)
    
    for edge_dim, description in scenarios:
        print(f"\n📊 场景: {description}")
        
        edge_attr = torch.randn(num_edges, edge_dim)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        try:
            with torch.no_grad():
                output = baseline_encoder(data)
            print(f"   ✅ 成功处理: 输入{edge_dim}维 → 输出{output.shape}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print(f"\n✅ 边特征维度处理测试完成")


if __name__ == "__main__":
    success = test_baseline_dimension_fix()
    if success:
        test_edge_dimension_scenarios()