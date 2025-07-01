# env/topology_loader.py

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
import random

def generate_realistic_backbone_topology(config: Dict) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    生成真实的骨干网络拓扑
    
    特点：
    1. 分层结构：Core -> Aggregation -> Edge
    2. 真实的资源异构性
    3. 动态负载和故障模拟
    4. 地理位置影响的延迟
    """
    
    # 网络层次配置
    core_nodes = 6      # 核心路由器
    agg_nodes = 12      # 汇聚路由器  
    edge_nodes = 24     # 边缘路由器
    total_nodes = core_nodes + agg_nodes + edge_nodes
    
    # 创建图
    G = nx.Graph()
    
    # 节点类型标记
    node_types = {}
    node_locations = {}  # 地理位置
    
    # 1. 添加核心层节点
    for i in range(core_nodes):
        G.add_node(i)
        node_types[i] = 'core'
        # 核心节点位置（模拟主要数据中心）
        node_locations[i] = generate_core_location(i)
    
    # 2. 添加汇聚层节点
    for i in range(core_nodes, core_nodes + agg_nodes):
        G.add_node(i)
        node_types[i] = 'aggregation'
        # 汇聚节点位置（区域数据中心）
        node_locations[i] = generate_aggregation_location(i - core_nodes)
    
    # 3. 添加边缘层节点
    for i in range(core_nodes + agg_nodes, total_nodes):
        G.add_node(i)
        node_types[i] = 'edge'
        # 边缘节点位置（边缘数据中心）
        node_locations[i] = generate_edge_location(i - core_nodes - agg_nodes)
    
    # 4. 构建真实的连接模式
    _add_realistic_connections(G, core_nodes, agg_nodes, edge_nodes, node_types)
    
    # 5. 生成异构节点特征
    node_features = _generate_heterogeneous_node_features(total_nodes, node_types, config)
    
    # 6. 生成真实的边特征（考虑地理距离和网络层次）
    edge_features = _generate_realistic_edge_features(G, node_types, node_locations, config)
    
    return G, node_features, edge_features

def generate_core_location(node_id: int) -> Tuple[float, float]:
    """生成核心节点的地理位置（主要城市）"""
    # 模拟全球主要数据中心位置
    major_cities = [
        (40.7128, -74.0060),   # 纽约
        (37.7749, -122.4194),  # 旧金山
        (51.5074, -0.1278),    # 伦敦
        (35.6762, 139.6503),   # 东京
        (52.5200, 13.4050),    # 柏林
        (39.9042, 116.4074)    # 北京
    ]
    return major_cities[node_id % len(major_cities)]

def generate_aggregation_location(node_id: int) -> Tuple[float, float]:
    """生成汇聚节点位置（区域中心）"""
    # 在主要城市周围分布
    base_locations = [
        (41.8781, -87.6298),   # 芝加哥
        (34.0522, -118.2437),  # 洛杉矶
        (48.8566, 2.3522),     # 巴黎
        (35.6895, 139.6917),   # 东京周边
        (53.5511, 9.9937),     # 汉堡
        (31.2304, 121.4737)    # 上海
    ]
    # 添加随机偏移
    base_lat, base_lon = base_locations[node_id % len(base_locations)]
    lat_offset = np.random.uniform(-2, 2)
    lon_offset = np.random.uniform(-2, 2)
    return (base_lat + lat_offset, base_lon + lon_offset)

def generate_edge_location(node_id: int) -> Tuple[float, float]:
    """生成边缘节点位置（本地接入点）"""
    # 更分散的位置分布
    lat = np.random.uniform(25, 60)  # 北纬25-60度
    lon = np.random.uniform(-130, 140)  # 经度范围
    return (lat, lon)

def _add_realistic_connections(G: nx.Graph, core_nodes: int, agg_nodes: int, 
                              edge_nodes: int, node_types: Dict):
    """添加真实的网络连接"""
    
    # 1. 核心层全连接（高冗余）
    for i in range(core_nodes):
        for j in range(i + 1, core_nodes):
            G.add_edge(i, j, connection_type='core_core')
    
    # 2. 核心层到汇聚层（每个汇聚节点连接多个核心节点）
    for agg_id in range(core_nodes, core_nodes + agg_nodes):
        # 每个汇聚节点连接2-3个核心节点（冗余）
        core_connections = np.random.choice(
            range(core_nodes), 
            size=np.random.randint(2, 4), 
            replace=False
        )
        for core_id in core_connections:
            G.add_edge(core_id, agg_id, connection_type='core_agg')
    
    # 3. 汇聚层内部连接（部分连接）
    agg_start = core_nodes
    agg_end = core_nodes + agg_nodes
    for i in range(agg_start, agg_end):
        for j in range(i + 1, agg_end):
            # 30%概率连接（模拟区域互联）
            if np.random.random() < 0.3:
                G.add_edge(i, j, connection_type='agg_agg')
    
    # 4. 汇聚层到边缘层
    edge_start = core_nodes + agg_nodes
    edge_end = core_nodes + agg_nodes + edge_nodes
    
    for edge_id in range(edge_start, edge_end):
        # 每个边缘节点连接1-2个汇聚节点
        agg_connections = np.random.choice(
            range(agg_start, agg_end),
            size=np.random.randint(1, 3),
            replace=False
        )
        for agg_id in agg_connections:
            G.add_edge(agg_id, edge_id, connection_type='agg_edge')
    
    # 5. 边缘层部分互联（模拟本地连接）
    for i in range(edge_start, edge_end):
        for j in range(i + 1, edge_end):
            # 10%概率连接（模拟边缘互联）
            if np.random.random() < 0.1:
                G.add_edge(i, j, connection_type='edge_edge')

def _generate_heterogeneous_node_features(total_nodes: int, node_types: Dict, config: Dict) -> np.ndarray:
    """
    生成异构节点特征
    
    特征维度：[CPU, Memory, Storage, Network_Capacity, Load, Reliability, Cost, Latency_Sensitivity]
    """
    
    features = np.zeros((total_nodes, 8))
    
    for node_id in range(total_nodes):
        node_type = node_types[node_id]
        
        if node_type == 'core':
            # 核心节点：高性能、高可靠性、高成本
            features[node_id] = [
                np.random.uniform(0.8, 1.0),    # CPU
                np.random.uniform(0.8, 1.0),    # Memory
                np.random.uniform(0.7, 1.0),    # Storage
                np.random.uniform(0.9, 1.0),    # Network_Capacity
                np.random.uniform(0.2, 0.6),    # Current_Load
                np.random.uniform(0.95, 1.0),   # Reliability
                np.random.uniform(0.8, 1.0),    # Cost
                np.random.uniform(0.9, 1.0)     # Latency_Sensitivity
            ]
        elif node_type == 'aggregation':
            # 汇聚节点：中等性能
            features[node_id] = [
                np.random.uniform(0.6, 0.8),    # CPU
                np.random.uniform(0.6, 0.8),    # Memory
                np.random.uniform(0.5, 0.8),    # Storage
                np.random.uniform(0.7, 0.9),    # Network_Capacity
                np.random.uniform(0.3, 0.7),    # Current_Load
                np.random.uniform(0.85, 0.95),  # Reliability
                np.random.uniform(0.5, 0.7),    # Cost
                np.random.uniform(0.7, 0.9)     # Latency_Sensitivity
            ]
        else:  # edge
            # 边缘节点：资源有限、成本敏感
            features[node_id] = [
                np.random.uniform(0.3, 0.6),    # CPU
                np.random.uniform(0.3, 0.6),    # Memory
                np.random.uniform(0.2, 0.6),    # Storage
                np.random.uniform(0.4, 0.7),    # Network_Capacity
                np.random.uniform(0.4, 0.8),    # Current_Load
                np.random.uniform(0.7, 0.9),    # Reliability
                np.random.uniform(0.2, 0.5),    # Cost
                np.random.uniform(0.5, 0.8)     # Latency_Sensitivity
            ]
        
        # 添加现实的噪声和动态性
        noise = np.random.normal(0, 0.05, 8)
        features[node_id] += noise
        features[node_id] = np.clip(features[node_id], 0.01, 1.0)
    
    return features

def _generate_realistic_edge_features(G: nx.Graph, node_types: Dict, 
                                    node_locations: Dict, config: Dict) -> np.ndarray:
    """
    生成真实的边特征
    
    特征维度：[Available_Bandwidth, Current_Latency, Jitter, Packet_Loss, 
              Utilization, Congestion_Level, Failure_Rate, QoS_Class]
    """
    
    edges = list(G.edges())
    edge_features = np.zeros((len(edges), 8))
    
    for idx, (u, v) in enumerate(edges):
        connection_type = G.edges[u, v].get('connection_type', 'unknown')
        
        # 计算地理距离影响的基础延迟
        lat1, lon1 = node_locations[u]
        lat2, lon2 = node_locations[v]
        geo_distance = _calculate_distance(lat1, lon1, lat2, lon2)
        base_latency = max(1.0, geo_distance / 200000)  # 光速传播延迟
        
        # 根据连接类型设置参数
        if connection_type == 'core_core':
            # 核心间链路：高带宽、低延迟、高质量
            bandwidth = np.random.uniform(80, 100)
            latency = base_latency + np.random.uniform(0.1, 0.5)
            jitter = np.random.uniform(0.01, 0.05)
            packet_loss = np.random.uniform(0.0001, 0.001)
            utilization = np.random.uniform(0.3, 0.7)
            congestion = np.random.uniform(0.1, 0.4)
            failure_rate = np.random.uniform(0.001, 0.01)
            qos_class = np.random.uniform(0.9, 1.0)
            
        elif connection_type == 'core_agg':
            # 核心到汇聚：高带宽、中等延迟
            bandwidth = np.random.uniform(60, 90)
            latency = base_latency + np.random.uniform(0.5, 1.5)
            jitter = np.random.uniform(0.02, 0.08)
            packet_loss = np.random.uniform(0.001, 0.005)
            utilization = np.random.uniform(0.4, 0.8)
            congestion = np.random.uniform(0.2, 0.6)
            failure_rate = np.random.uniform(0.01, 0.02)
            qos_class = np.random.uniform(0.8, 0.95)
            
        elif connection_type == 'agg_edge':
            # 汇聚到边缘：中等带宽、较高延迟
            bandwidth = np.random.uniform(30, 70)
            latency = base_latency + np.random.uniform(1.0, 3.0)
            jitter = np.random.uniform(0.05, 0.15)
            packet_loss = np.random.uniform(0.005, 0.02)
            utilization = np.random.uniform(0.5, 0.9)
            congestion = np.random.uniform(0.3, 0.8)
            failure_rate = np.random.uniform(0.02, 0.05)
            qos_class = np.random.uniform(0.6, 0.85)
            
        else:  # 其他连接类型
            # 默认参数
            bandwidth = np.random.uniform(20, 60)
            latency = base_latency + np.random.uniform(2.0, 5.0)
            jitter = np.random.uniform(0.1, 0.3)
            packet_loss = np.random.uniform(0.01, 0.05)
            utilization = np.random.uniform(0.6, 0.95)
            congestion = np.random.uniform(0.4, 0.9)
            failure_rate = np.random.uniform(0.03, 0.08)
            qos_class = np.random.uniform(0.4, 0.7)
        
        edge_features[idx] = [
            bandwidth, latency, jitter, packet_loss,
            utilization, congestion, failure_rate, qos_class
        ]
    
    return edge_features

def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的地理距离（米）"""
    from math import radians, cos, sin, asin, sqrt
    
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # 地球半径（米）
    
    return c * r

def simulate_dynamic_network_conditions(G: nx.Graph, edge_features: np.ndarray, 
                                      time_step: int) -> np.ndarray:
    """
    模拟动态网络条件
    
    包括：
    1. 流量波动
    2. 链路故障
    3. 拥塞变化
    4. QoS降级
    """
    
    # 复制原始特征
    dynamic_features = edge_features.copy()
    
    # 1. 模拟流量的日周期变化
    daily_cycle = np.sin(2 * np.pi * time_step / 24) * 0.3 + 1.0
    
    # 2. 随机流量尖峰
    traffic_spikes = np.random.exponential(0.1, len(dynamic_features))
    
    # 3. 链路故障模拟（5%概率）
    failure_mask = np.random.random(len(dynamic_features)) < 0.05
    
    for idx in range(len(dynamic_features)):
        # 更新带宽（受流量影响）
        traffic_factor = daily_cycle * (1 + traffic_spikes[idx])
        dynamic_features[idx, 0] *= (1 / traffic_factor)
        
        # 更新延迟（拥塞影响）
        congestion_multiplier = 1 + traffic_spikes[idx] * 2
        dynamic_features[idx, 1] *= congestion_multiplier
        
        # 更新抖动
        dynamic_features[idx, 2] *= (1 + traffic_spikes[idx])
        
        # 链路故障处理
        if failure_mask[idx]:
            dynamic_features[idx, 0] *= 0.1  # 带宽严重下降
            dynamic_features[idx, 1] *= 5.0  # 延迟大幅增加
            dynamic_features[idx, 3] *= 10.0 # 丢包率增加
        
        # 更新利用率
        dynamic_features[idx, 4] = min(0.99, dynamic_features[idx, 4] * traffic_factor)
        
        # 更新拥塞级别
        dynamic_features[idx, 5] = min(1.0, dynamic_features[idx, 5] + traffic_spikes[idx] * 0.5)
    
    # 确保数值范围合理
    dynamic_features = np.clip(dynamic_features, 0.001, 1000)
    
    return dynamic_features

# 兼容原有接口的函数
def generate_topology(num_nodes=20, prob=0.3):
    """兼容原有的简单拓扑生成（用于测试）"""
    G = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=None, directed=False)
    
    # 简单的节点特征
    node_features = np.random.rand(len(G.nodes), 8)
    
    # 简单的边特征  
    edge_features = []
    for u, v in G.edges():
        bandwidth = np.random.uniform(10, 100)
        delay = np.random.uniform(1, 10)
        jitter = np.random.uniform(0, 2)
        loss = np.random.uniform(0, 0.1)
        utilization = np.random.uniform(0.2, 0.8)
        congestion = np.random.uniform(0.1, 0.7)
        failure_rate = np.random.uniform(0.01, 0.05)
        qos_class = np.random.uniform(0.5, 1.0)
        
        edge_features.append([bandwidth, delay, jitter, loss, utilization, congestion, failure_rate, qos_class])
    
    return G, node_features, np.array(edge_features)