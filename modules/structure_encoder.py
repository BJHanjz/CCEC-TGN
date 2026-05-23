"""
StreamingGraphEngine:维护每个节点当前的结构分数，并量化成整数 k_core 风格特征。
CausalHardNegativeSampler:用这些结构分数采结构相似的负样本，让训练更难。
StructureEncoder:把这个整数结构特征编码成向量，变成神经网络能直接融合的表示。
"""
import torch
import torch.nn as nn
import numpy as np


class StructureEncoder(nn.Module):
    def __init__(self, embed_dim, max_core=200):
        super().__init__()
        # 使用 Embedding 编码离散的 Core 值
        self.core_embed = nn.Embedding(max_core + 1, embed_dim)
        # 简单的 MLP 进行特征变换
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, core_values):
        # core_values: [Batch]
        # 截断防止越界
        core_values = torch.clamp(core_values, 0, self.core_embed.num_embeddings - 1)
        emb = self.core_embed(core_values)
        return self.mlp(emb)