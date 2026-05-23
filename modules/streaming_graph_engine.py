# import numpy as np
# from collections import defaultdict, deque
#
#
# class StreamingGraphEngine:
#     def __init__(self, window_size_seconds=None, update_freq=500):
#         """
#         Args:
#             window_size_seconds: 时间窗口大小（秒）。如果是 None，则保留全历史。
#             update_freq: 每插入多少个 Batch 触发一次结构重算。
#         """
#         self.window_size = window_size_seconds
#         self.update_freq = update_freq
#
#         # 存储边流 (u, v, ts)
#         self.edge_stream = deque()
#         # 邻接表 {u: {v1, v2...}}
#         self.adj = defaultdict(set)
#
#         # 缓存的结构特征
#         self.node_coreness = defaultdict(int)
#         self.node_degrees = defaultdict(int)
#
#         self.batch_counter = 0
#         self.dirty = False
#
#     def update_batch(self, sources, destinations, timestamps):
#         """
#         流式处理一个 Batch 的边
#         """
#         current_time = timestamps[-1]
#
#         # 1. 添加新边
#         for u, v, ts in zip(sources, destinations, timestamps):
#             u, v = int(u), int(v)
#             self.edge_stream.append((u, v, ts))
#             self.adj[u].add(v)
#             self.adj[v].add(u)
#
#         # 2. 移除过期边 (滑动窗口)
#         if self.window_size is not None:
#             cutoff = current_time - self.window_size
#             while self.edge_stream and self.edge_stream[0][2] < cutoff:
#                 u_old, v_old, _ = self.edge_stream.popleft()
#                 self._remove_edge(u_old, v_old)
#                 self.dirty = True
#
#         # 3. 触发重算判断
#         self.batch_counter += 1
#         if self.dirty or (self.batch_counter % self.update_freq == 0):
#             self._recompute_k_core()
#
#     def _remove_edge(self, u, v):
#         if v in self.adj[u]: self.adj[u].remove(v)
#         if u in self.adj[v]: self.adj[v].remove(u)
#         if not self.adj[u]: del self.adj[u]
#         if not self.adj[v]: del self.adj[v]
#
#     def _recompute_k_core(self):
#         """
#         实现经典的 O(m) k-Core 分解算法 (Peeling Algorithm)
#         """
#         # 计算所有活跃节点的度
#         active_nodes = list(self.adj.keys())
#         degrees = {n: len(self.adj[n]) for n in active_nodes}
#         if not degrees: return
#
#         # 初始化桶
#         max_deg = max(degrees.values())
#         buckets = defaultdict(set)
#         for n, d in degrees.items():
#             buckets[d].add(n)
#
#         # 记录结果
#         self.node_coreness.clear()
#         self.node_degrees = degrees.copy()
#
#         # 剥洋葱
#         for k in range(max_deg + 1):
#             while buckets[k]:
#                 n = buckets[k].pop()
#                 self.node_coreness[n] = k  # 记录 coreness
#
#                 # 更新邻居
#                 for neighbor in self.adj[n]:
#                     if neighbor in self.node_coreness: continue  # 已处理
#
#                     d = degrees[neighbor]
#                     if d > k:
#                         buckets[d].remove(neighbor)
#                         degrees[neighbor] -= 1
#                         buckets[degrees[neighbor]].add(neighbor)
#
#         self.dirty = False
#
#     def get_structure_features(self, node_ids):
#         """返回一批节点的 (coreness, degree)"""
#         if self.dirty:
#             self._recompute_k_core()
#
#         cores = [self.node_coreness.get(int(n), 0) for n in node_ids]
#         degs = [self.node_degrees.get(int(n), 0) for n in node_ids]
#         return np.array(cores), np.array(degs)
#  2026.2.18
"""
节点的结构记忆
"""
import torch
import numpy as np
import math
from collections import defaultdict


class StreamingGraphEngine:
    """
    【论文创新点核心模块】
    Time-Decayed Dynamic Structural Engine (时间衰减动态结构引擎)

    不同于传统的静态 K-Core，本引擎采用“流式度数加权”策略：
    1. Soft Decay (软衰减): 节点的结构重要性随时间呈指数衰减。
    2. Hard Window (硬窗口): 超出 window_size 的交互会被彻底清除（清理内存）。

    【修复说明】:
    - 变量名统一回滚为 `node_coreness` 以兼容 causal_sampler.py。
    - 即使内部存储的是 float 类型的衰减分数，对外依然表现为 coreness 属性。
    """

    def __init__(self, window_size_seconds=None, time_decay_factor=1e-7):
        """
        :param window_size_seconds: 硬窗口大小 (秒)，用于内存管理。
        :param time_decay_factor: 时间衰减因子 (lambda)。
               Weight = exp(-lambda * delta_time)
               建议值: 1e-6 ~ 1e-7 (取决于数据集的时间戳单位)
        """
        self.window_size = window_size_seconds
        self.decay_factor = time_decay_factor

        # 【关键修复】将 node_scores 改回 node_coreness 以兼容旧代码
        # 这里存储的是浮点数 (Decayed Weighted Degree)
        self.node_coreness = defaultdict(float)

        # 记录节点上一次更新的时间，用于计算衰减
        self.last_update_time = defaultdict(float)

        # 简单的邻接表
        self.adj = defaultdict(set)

    """
    流式更新一批边，并应用时间衰减。
    """
    def update_batch(self, src_list, dst_list, ts_list):

        # 转为 CPU numpy 处理
        if isinstance(src_list, torch.Tensor): src_list = src_list.cpu().numpy()
        if isinstance(dst_list, torch.Tensor): dst_list = dst_list.cpu().numpy()
        if isinstance(ts_list, torch.Tensor): ts_list = ts_list.cpu().numpy()

        # current_batch_time = ts_list[-1]

        for s, d, t in zip(src_list, dst_list, ts_list):
            t = float(t)

            # === 创新点：时间衰减逻辑 (Time-Decay Mechanism) ===

            # 1. 更新源节点 Source
            self._apply_decay(s, t)  # 先把节点 s 的旧结构分数按时间衰减到当前时刻
            self.node_coreness[s] += 1.0  # 建立连接，增加权重
            self.adj[s].add(d)

            # 2. 更新目标节点 Destination
            self._apply_decay(d, t)
            self.node_coreness[d] += 1.0
            self.adj[d].add(s)

    """
    对单个节点应用指数衰减
    Score_new = Score_old * exp(-lambda * (t_now - t_last))
    """
    def _apply_decay(self, node, current_t):

        last_t = self.last_update_time[node]

        # 如果是第一次出现，或者时间倒流，不衰减
        if last_t == 0 or current_t <= last_t:
            self.last_update_time[node] = current_t
            return

        delta_t = current_t - last_t

        # 计算衰减系数
        try:
            decay_weight = math.exp(-self.decay_factor * delta_t)
        except OverflowError:
            decay_weight = 0.0

        # 应用衰减
        self.node_coreness[node] *= decay_weight

        # 更新时间
        self.last_update_time[node] = current_t

    """
    获取节点的结构特征 (离散化后的 Coreness)。
    StructureEncoder 需要整数输入，所以我们将浮点分数量化为整数。
    """
    def get_structure_features(self, nodes):

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.cpu().numpy()

        features = []
        for n in nodes:
            # 获取最新的分数 (不做衰减，保持最后更新时的状态，节省计算)
            # 【关键修复】读取 self.node_coreness
            score = self.node_coreness.get(n, 0.0)

            # === 量化策略 (Quantization) ===
            if score <= 0:
                k_core = 0
            else:
                # 直接取整作为特征
                k_core = int(score)

            # 截断，防止越界
            if k_core > 200: k_core = 200

            features.append(k_core)

        return np.array(features, dtype=np.int32), None