# 文件路径: modules/ctgn.py (最终“在线评估”版)

import torch
import torch.nn as nn
import numpy as np
from model.tgn import TGN


class CTGN(TGN):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers,
                 n_heads, dropout, use_memory, memory_dimension, embedding_module_type,
                 message_function, aggregator_type, memory_updater_type,
                 # 【架构升级】接收全局社群总数
                 num_communities,
                 **kwargs):

        # 将 TGN 认识的参数，以及 **kwargs (用于时间统计等) 传递给 super()
        # num_communities 是我们自己独有的，不能传给父类
        super(CTGN, self).__init__(neighbor_finder=neighbor_finder,
                                   node_features=node_features,
                                   edge_features=edge_features,
                                   device=device,
                                   n_layers=n_layers,
                                   n_heads=n_heads,
                                   dropout=dropout,
                                   use_memory=use_memory,
                                   memory_dimension=memory_dimension,
                                   embedding_module_type=embedding_module_type,
                                   message_function=message_function,
                                   aggregator_type=aggregator_type,
                                   memory_updater_type=memory_updater_type,
                                   **kwargs)

        self.community_embedding_dim = memory_dimension
        # 用于方案B：暂存“融合前”的核心嵌入
        self.last_base_embeddings = None
        # 用于方案A或最终预测：暂存“融合后”的嵌入
        self.last_fused_embeddings = None

        # 【架构升级】使用可学习的 nn.Embedding 层来表示社群
        self.community_embedding_layer = nn.Embedding(num_embeddings=num_communities,
                                                      embedding_dim=self.community_embedding_dim).to(device)

        # 门控网络保持不变
        # 作用不是简单把两种向量拼起来，而是学一个门控系数：𝛼∈(0,1)然后决定：更依赖 TGN 的基础时序表示,还是更依赖社群表示
        self.gate_network = nn.Sequential(
            nn.Linear(self.community_embedding_dim * 2, self.community_embedding_dim),
            nn.ReLU(), nn.Linear(self.community_embedding_dim, 1), nn.Sigmoid()).to(device)

    def get_community_embeddings(self, nodes, partition):
        """
        【架构升级】通过查 nn.Embedding 词典的方式，获取社群嵌入
        1.根据 partition 查每个节点的社群 ID
        2.对找不到的节点默认给 0
        3.去 embedding 表里取对应社群向量
        """
        # 如果节点不在当前快照的社群划分中（例如新节点），给它一个默认的社群ID 0
        # 这个默认ID 0 对应的嵌入向量也会在训练中被学习
        community_ids = [partition.get(str(node_id), 0) for node_id in nodes]
        community_ids_torch = torch.tensor(community_ids, dtype=torch.long, device=self.device)

        return self.community_embedding_layer(community_ids_torch)

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                edge_idxs, n_neighbors=None, partition=None):
        """
        这个 forward 函数主要用于训练，它采用“批处理”模式。
        """
        # Step 1: 获取“核心引擎”的产出 (base embeddings)
        source_embedding, destination_embedding, negative_embedding = \
            self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes,
                                             edge_times, edge_idxs, n_neighbors)

        # 【理论升华】暂存“融合前”的核心嵌入，用于方案B的正则化
        self.last_base_embeddings = (source_embedding, destination_embedding)

        # 如果不提供社群信息 (例如 warmup 阶段或 lamda=0)，则表现为纯TGN
        if partition is None:
            pos_score = self.affinity_score(source_embedding, destination_embedding).squeeze(dim=-1)
            neg_score = self.affinity_score(source_embedding, negative_embedding).squeeze(dim=-1)
            # 保持接口一致性，将未融合的嵌入存入fused暂存器
            self.last_fused_embeddings = (source_embedding, destination_embedding)
            return pos_score, neg_score

        # Step 2: 进行社群融合
        source_community_embedding = self.get_community_embeddings(source_nodes, partition)
        destination_community_embedding = self.get_community_embeddings(destination_nodes, partition)
        negative_community_embedding = self.get_community_embeddings(negative_nodes, partition)

        # 对源、目标、负样本分别进行门控融合
        gate_input_src = torch.cat([source_embedding, source_community_embedding], dim=1)
        gate_alpha_src = self.gate_network(gate_input_src)
        final_source_embedding = gate_alpha_src * source_embedding + (1 - gate_alpha_src) * source_community_embedding

        gate_input_dst = torch.cat([destination_embedding, destination_community_embedding], dim=1)
        gate_alpha_dst = self.gate_network(gate_input_dst)
        final_destination_embedding = gate_alpha_dst * destination_embedding + (
                1 - gate_alpha_dst) * destination_community_embedding

        gate_input_neg = torch.cat([negative_embedding, negative_community_embedding], dim=1)
        gate_alpha_neg = self.gate_network(gate_input_neg)
        final_negative_embedding = gate_alpha_neg * negative_embedding + (
                1 - gate_alpha_neg) * negative_community_embedding

        # 暂存“融合后”的嵌入
        self.last_fused_embeddings = (final_source_embedding, final_destination_embedding)

        # Step 3: 计算最终得分
        # 用融合后的 embedding 做链路预测,训练目标最终看的不是 base embedding,而是 fused embedding
        # 所以 community 信息不是一个辅助 loss，也不是一个 side feature，而是直接进入最终预测表示
        pos_score = self.affinity_score(final_source_embedding, final_destination_embedding).squeeze(dim=-1)
        neg_score = self.affinity_score(final_source_embedding, final_negative_embedding).squeeze(dim=-1)

        return pos_score, neg_score

    def compute_edge_probabilities_online(self, source_nodes, destination_nodes, negative_nodes,
                                          edge_times, edge_idxs, n_neighbors, partition=None):
        """
        【新增接口】一个专门用于“在线”评估的函数，严格遵循 TGN 官方的评估协议。
        它通过调用父类的 compute_temporal_embeddings，隐式地实现了“先用正样本更新内存，再计算负样本嵌入”的逻辑。
        """
        # Step 1: 调用父类核心函数，获取三组基础嵌入。
        # 这一步内部已经隐式地、序列化地更新了内存！
        source_embedding, destination_embedding, negative_embedding = \
            self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes,
                                             edge_times, edge_idxs, n_neighbors)

        # 如果没有社群信息，直接用基础嵌入计算得分并返回
        if partition is None:
            pos_prob = self.affinity_score(source_embedding, destination_embedding).squeeze(dim=-1)
            neg_prob = self.affinity_score(source_embedding, negative_embedding).squeeze(dim=-1)
            return pos_prob, neg_prob

        # Step 2: 如果有社群信息，则对三组基础嵌入进行融合
        source_community_embedding = self.get_community_embeddings(source_nodes, partition)
        destination_community_embedding = self.get_community_embeddings(destination_nodes, partition)
        negative_community_embedding = self.get_community_embeddings(negative_nodes, partition)

        gate_input_src = torch.cat([source_embedding, source_community_embedding], dim=1)
        gate_alpha_src = self.gate_network(gate_input_src)
        fused_source_embedding = gate_alpha_src * source_embedding + (1 - gate_alpha_src) * source_community_embedding

        gate_input_dst = torch.cat([destination_embedding, destination_community_embedding], dim=1)
        gate_alpha_dst = self.gate_network(gate_input_dst)
        fused_destination_embedding = gate_alpha_dst * destination_embedding + (
                1 - gate_alpha_dst) * destination_community_embedding

        gate_input_neg = torch.cat([negative_embedding, negative_community_embedding], dim=1)
        gate_alpha_neg = self.gate_network(gate_input_neg)
        fused_negative_embedding = gate_alpha_neg * negative_embedding + (
                1 - gate_alpha_neg) * negative_community_embedding

        # Step 3: 用融合后的嵌入计算最终得分
        pos_prob = self.affinity_score(fused_source_embedding, fused_destination_embedding).squeeze(dim=-1)
        neg_prob = self.affinity_score(fused_source_embedding, fused_negative_embedding).squeeze(dim=-1)

        return pos_prob, neg_prob