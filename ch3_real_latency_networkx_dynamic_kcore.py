import sys
import os
import time
from collections import deque, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_processing import get_data
from modules.streaming_graph_engine import StreamingGraphEngine

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


datasets = ['wikipedia', 'mooc', 'reddit', 'lastfm']
display_names = ['Wikipedia', 'MOOC', 'Reddit', 'LastFM']
WINDOW_SIZE_SECONDS = 259200
BATCH_SIZE = 100

ours_time_ms = []  # 本文方法平均每条边耗时多少毫秒
dynamic_kcore_time_ms = []  # 你的方法平均每条边耗时多少毫秒
static_time_ms = []  # 动态 k-core baseline 平均每条边耗时多少毫秒


class SlidingWindowSimpleGraph:
    """
    维护滑窗上的“简单图 + 重复边计数”
    """
    def __init__(self, window_size_seconds: int):
        self.window_size_seconds = window_size_seconds
        self.graph = nx.Graph()
        self.edge_queue = deque()
        self.edge_mult = defaultdict(int)

    @staticmethod
    def _norm_edge(u, v):
        return (u, v) if u <= v else (v, u)  # 因为这是无向图，所以 (u, v) 和 (v, u) 视为同一条边。

    def add_edge(self, u, v, ts):
        key = self._norm_edge(u, v)
        self.edge_queue.append((u, v, ts))
        self.edge_mult[key] += 1
        if self.edge_mult[key] == 1:
            self.graph.add_edge(u, v)
            return True
        return False

    """
    负责滑动窗口过期逻辑：
    把早于 current_ts - window_size_seconds 的边事件从队列左端弹出
    对应边计数减 1
    若计数减到 0，说明这条简单边在窗口中彻底消失，于是从 graph 删除
    """
    def expire_until(self, current_ts):
        removed_simple_edges = []
        while self.edge_queue and (current_ts - self.edge_queue[0][2] > self.window_size_seconds):
            u, v, _ = self.edge_queue.popleft()
            key = self._norm_edge(u, v)
            self.edge_mult[key] -= 1
            if self.edge_mult[key] == 0:
                del self.edge_mult[key]
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    removed_simple_edges.append((u, v))
        return removed_simple_edges


class DynamicKCoreMaintenance:
    """
    更快版 dynamic k-core baseline：

    改进点：
    1) repair 低频触发，而不是每个 batch 都触发
    2) repair patch 从二跳改成一跳
    3) seeds 太大时跳过 repair，避免 patch 爆炸
    """

    def __init__(
        self,
        window_size_seconds: int,
        repair_interval: int = 20,
        max_repair_seeds: int = 64,
        enable_repair: bool = True,
        verbose_debug: bool = False,
    ):
        self.sw_graph = SlidingWindowSimpleGraph(window_size_seconds)
        self.graph = self.sw_graph.graph
        self.core = {}

        self.repair_interval = repair_interval
        self.max_repair_seeds = max_repair_seeds
        self.enable_repair = enable_repair
        self.verbose_debug = verbose_debug
        self.batch_count = 0

    # 如果一个节点还没出现在 self.core 里，就给它初始化为 0。这是维护 core 字典的基本操作。
    def _ensure_node(self, u):
        if u not in self.core:
            self.core[u] = 0

    # 计算节点 u 有多少个邻居，其 core 值至少为 k，即支撑度
    def _support_ge(self, u, k):
        if u not in self.graph:
            return 0
        cnt = 0
        for v in self.graph.neighbors(u):
            if self.core.get(v, 0) >= k:
                cnt += 1
        return cnt

    def _collect_insert_candidates(self, seeds):
        """
        插入时候选区：
        只扩 1 层，避免 BFS 扩张太大
        只保留那些 core 值和种子差不多的邻居
        """
        candidate = set()
        for u in seeds:
            if u not in self.graph:
                continue
            candidate.add(u)
            ku = self.core.get(u, 0)
            for v in self.graph.neighbors(u):
                kv = self.core.get(v, 0)
                if abs(kv - ku) <= 1:
                    candidate.add(v)
        return candidate

    """
    收集候选区,对候选节点反复检查：
    如果它当前 core 是 ku
    且它有至少 ku + 1 个邻居的 core 不小于 ku + 1
    那它的 core 就可以 +1
    如果某个节点升核了，再继续激活周围节点，直到稳定
    """
    def _try_incremental_raise(self, seeds):
        candidate = self._collect_insert_candidates(seeds)
        if not candidate:
            return set()

        changed = set()
        active = set(candidate)

        while active:
            next_active = set()
            for u in active:
                if u not in self.graph:
                    continue
                ku = self.core.get(u, 0)
                if self._support_ge(u, ku + 1) >= ku + 1:
                    self.core[u] = ku + 1
                    changed.add(u)
                    next_active.add(u)
                    for v in self.graph.neighbors(u):
                        if v in candidate:
                            next_active.add(v)
            active = next_active

        return changed

    """
    从删除边涉及的端点出发，做一个 BFS 式扫描，找出“不稳定节点”(support_ge(u, ku) < ku)。
    """
    def _find_unstable_nodes_after_deletion(self, seeds):
        unstable = set()
        q = deque(seeds)
        seen = set(seeds)

        while q:
            u = q.popleft()
            if u not in self.graph:
                continue
            ku = self.core.get(u, 0)
            if ku > 0 and self._support_ge(u, ku) < ku:
                unstable.add(u)
                for v in self.graph.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
        return unstable

    """
    到不稳定节点后，就要做“级联降核”：
    如果节点当前 core 过高且不满足支持条件，就减 1
    节点一旦降核，它的邻居也可能因此失去支撑，于是继续进队列
    如此反复直到所有节点稳定
    """
    def _cascade_decrease(self, seeds):
        changed = set()
        q = deque(seeds)
        in_queue = set(seeds)

        while q:
            u = q.popleft()
            in_queue.discard(u)

            if u not in self.graph:
                continue

            while self.core.get(u, 0) > 0 and self._support_ge(u, self.core[u]) < self.core[u]:
                self.core[u] -= 1
                changed.add(u)
                for v in self.graph.neighbors(u):
                    if v not in in_queue:
                        q.append(v)
                        in_queue.add(v)

        return changed

    def _local_repair(self, seeds):
        """
        一跳 patch 修复，比原来的二跳快很多
        从 seeds 出发，只取一跳 patch
        在这个 patch 诱导子图上，直接调用 nx.core_number(subg)
        用这个局部精确结果去修补 self.core

        因为前面的增量升核/降核都是启发式的，可能积累误差。所以需要定期做一个“局部精修”。
        """
        if not seeds:
            return

        patch = set()
        for u in seeds:
            if u not in self.graph:
                continue
            patch.add(u)
            for v in self.graph.neighbors(u):
                patch.add(v)

        if not patch:
            return

        subg = self.graph.subgraph(patch)

        if self.verbose_debug:
            print(
                f"[repair] seeds={len(seeds)}, "
                f"patch_nodes={len(patch)}, patch_edges={subg.number_of_edges()}, "
                f"graph_nodes={self.graph.number_of_nodes()}, graph_edges={self.graph.number_of_edges()}"
            )

        if subg.number_of_nodes() == 0:
            return
        elif subg.number_of_edges() == 0:
            repaired = {n: 0 for n in subg.nodes()}
        else:
            repaired = nx.core_number(subg)

        for n in patch:
            self.core[n] = repaired.get(n, 0)

    """
    动态 k-core baseline 的主入口:
    第一步：插入 batch 中的新边,维护滑窗图,记录哪些端点真的引起了简单图变化
    第二步：窗口过期,把过期边删掉,记录哪些边删除引起了图结构变化
    第三步：清理已经不在图中的节点,如果一个节点在 core 字典里，但图里已经没它了，就删掉
    第四步：增量升核,对插入导致的变化尝试局部 raise
    第五步：级联降核,对删除导致的变化尝试 cascade decrease
    第六步：低频局部修复
    如果满足条件，就做 _local_repair,最后返回 self.core。
    """
    def process_batch(self, src_batch, dst_batch, ts_batch):
        self.batch_count += 1

        inserted_endpoints = set()
        deleted_endpoints = set()
        structure_changed_nodes = set()

        # 1) 插入 batch
        for u, v, ts in zip(src_batch, dst_batch, ts_batch):
            self._ensure_node(u)
            self._ensure_node(v)
            changed = self.sw_graph.add_edge(u, v, ts)
            if changed:
                inserted_endpoints.add(u)
                inserted_endpoints.add(v)
                structure_changed_nodes.add(u)
                structure_changed_nodes.add(v)

        # 2) 滑窗过期
        current_ts = ts_batch[-1]
        removed_edges = self.sw_graph.expire_until(current_ts)
        for u, v in removed_edges:
            deleted_endpoints.add(u)
            deleted_endpoints.add(v)
            structure_changed_nodes.add(u)
            structure_changed_nodes.add(v)

        # 3) 清理已不在图中的节点
        for n in list(structure_changed_nodes):
            if n in self.core and n not in self.graph:
                del self.core[n]

        # 4) 增量升核
        raise_changed = set()
        if inserted_endpoints:
            raise_changed = self._try_incremental_raise(inserted_endpoints)

        # 5) 级联降核
        lower_changed = set()
        if deleted_endpoints:
            unstable = self._find_unstable_nodes_after_deletion(deleted_endpoints)
            if unstable:
                lower_changed = self._cascade_decrease(unstable)

        # 6) 低频局部修复
        repair_seeds = inserted_endpoints | deleted_endpoints | raise_changed | lower_changed

        should_repair = (
            self.enable_repair
            and repair_seeds
            and len(repair_seeds) <= self.max_repair_seeds
            and self.batch_count % self.repair_interval == 0
        )

        if should_repair:
            self._local_repair(repair_seeds)

        return self.core


print("🚀 开始执行性能评测 (CTDS-Engine vs Dynamic k-core maintenance vs Static Recomputation)...")
"""
每个数据集的流程都一样：
1.读取完整数据流
2.截取前 10000 条边做评测
3.分别测三种方法
4.把平均耗时存下来
"""
for ds in datasets:
    _, _, full_data, _, _, _, _, _ = get_data(ds)
    test_size = min(10000, len(full_data.sources))
    src_data = full_data.sources[:test_size]
    dst_data = full_data.destinations[:test_size]
    ts_data = full_data.timestamps[:test_size]

    # 1) CTDS-Engine
    engine = StreamingGraphEngine(
        window_size_seconds=WINDOW_SIZE_SECONDS,
        time_decay_factor=1e-6
    )
    t0 = time.time()
    engine.update_batch(src_data, dst_data, ts_data)
    t1 = time.time()
    avg_ms_ours = ((t1 - t0) / test_size) * 1000
    ours_time_ms.append(avg_ms_ours)

    # 2) Dynamic k-core maintenance
    dynamic_baseline = DynamicKCoreMaintenance(
        window_size_seconds=WINDOW_SIZE_SECONDS,
        repair_interval=20,     # 每 20 个 batch 修一次
        max_repair_seeds=64,    # seeds 太大就不修
        enable_repair=True,
        verbose_debug=False
    )

    t0_dynamic = time.time()
    for i in range(0, test_size, BATCH_SIZE):
        start, end = i, min(i + BATCH_SIZE, test_size)
        dynamic_baseline.process_batch(
            src_data[start:end],
            dst_data[start:end],
            ts_data[start:end]
        )
    t1_dynamic = time.time()
    avg_ms_dynamic = ((t1_dynamic - t0_dynamic) / test_size) * 1000
    dynamic_kcore_time_ms.append(avg_ms_dynamic)

    # 3) Static recomputation
    static_sw = SlidingWindowSimpleGraph(window_size_seconds=WINDOW_SIZE_SECONDS)
    G = static_sw.graph

    t0_static = time.time()
    for i in range(0, test_size, BATCH_SIZE):
        start, end = i, min(i + BATCH_SIZE, test_size)

        for j in range(start, end):
            static_sw.add_edge(src_data[j], dst_data[j], ts_data[j])

        current_ts = ts_data[end - 1]
        static_sw.expire_until(current_ts)

        if G.number_of_nodes() == 0:
            _ = {}
        elif G.number_of_edges() == 0:
            _ = {n: 0 for n in G.nodes()}
        else:
            _ = nx.core_number(G)

    t1_static = time.time()
    avg_ms_static = ((t1_static - t0_static) / test_size) * 1000
    static_time_ms.append(avg_ms_static)

    print(
        f"[{ds}] "
        f"Ours={avg_ms_ours:.6f} ms/edge | "
        f"Dynamic k-core maintenance={avg_ms_dynamic:.6f} ms/edge | "
        f"Static={avg_ms_static:.6f} ms/edge"
    )

# ================= 绘图 =================
x = np.arange(len(datasets))
width = 0.24

fig, ax = plt.subplots(figsize=(9.5, 5.4))

all_vals = ours_time_ms + dynamic_kcore_time_ms + static_time_ms
positive_vals = [v for v in all_vals if v > 0]
safe_bottom = min(positive_vals) * 0.5 if positive_vals else 1e-4

rects1 = ax.bar(
    x - width,
    static_time_ms,
    width,
    label='Static Recomputation',
    color='#F67280',
    edgecolor='#C06C84',
    bottom=safe_bottom,
)
rects2 = ax.bar(
    x,
    dynamic_kcore_time_ms,
    width,
    label='Dynamic k-core Maintenance',
    color='#C7CEEA',
    edgecolor='#8E9AAF',
    bottom=safe_bottom,
)
rects3 = ax.bar(
    x + width,
    ours_time_ms,
    width,
    label='Ours',
    color='#A8D8EA',
    edgecolor='#88B8CA',
    bottom=safe_bottom,
)

ax.set_ylabel('Average Processing Time (ms/Edge)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=12)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

ax.set_yscale('log')
ax.set_ylim(safe_bottom, max(all_vals) * 10 if max(all_vals) > 0 else 1)

def autolabel(rects, decimals=4):
    for rect in rects:
        val = rect.get_height() + safe_bottom
        ax.annotate(
            f'{val:.{decimals}f}',
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
        )

autolabel(rects1, decimals=3)
autolabel(rects2, decimals=3)
autolabel(rects3, decimals=4)

ax.grid(axis='y', linestyle='-', alpha=0.3, color='#bbbbbb', which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('ch3_latency_with_dynamic_kcore.png', dpi=300, bbox_inches='tight')
print('✅ 已生成图表：ch3_latency_with_dynamic_kcore.png')