"""
选一个节点，沿着整条时间流不断更新结构引擎，然后对比“无时间衰减”和“有时间衰减”
两种结构分数的演化曲线，最后画图保存。
"""
import sys
import os

# 强制将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from utils.data_processing import get_data
from modules.streaming_graph_engine import StreamingGraphEngine  # 确保导入路径正确

# 解决 Matplotlib 中文字体问题（虽然此图已全英文，但保留有备无患）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载维基百科数据
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
    "wikipedia")

# 2. 挑选一个典型节点
target_node = 5
times = []
static_scores = []
decay_scores = []

# 3. 初始化两个引擎
engine_static = StreamingGraphEngine(window_size_seconds=259200, time_decay_factor=0.0)
engine_decay = StreamingGraphEngine(window_size_seconds=259200, time_decay_factor=1e-6)

# 4. 模拟时间流逝
print("正在模拟时间流逝并记录特征...")
for i in range(len(full_data.sources)):
    src, dst, ts = full_data.sources[i], full_data.destinations[i], full_data.timestamps[i]

    engine_static.update_batch(np.array([src]), np.array([dst]), np.array([ts]))
    engine_decay.update_batch(np.array([src]), np.array([dst]), np.array([ts]))

    if i % 1000 == 0:
        times.append(ts)
        static_scores.append(engine_static.node_coreness.get(target_node, 0))
        decay_scores.append(engine_decay.node_coreness.get(target_node, 0))

# 5. 画图
plt.figure(figsize=(9, 4.5))  # 稍微调整比例，更适合插入 Word

# 图例使用纯英文
plt.plot(times, static_scores, label="无衰减连续核心度", color='#d62728', linestyle='--', linewidth=2)
plt.plot(times, decay_scores, label="本章方法", color='#1f77b4', linewidth=2.5)
plt.fill_between(times, decay_scores, color='#1f77b4', alpha=0.15)

# 移除 plt.title()，全面精简坐标轴
plt.xlabel("Timestamp", fontsize=13)
plt.ylabel("Coreness", fontsize=13)

plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 保存为高质量 PNG
plt.savefig("ch3_feature_evolution_pure.png", dpi=300, bbox_inches='tight')
print("纯净版图表已成功保存为: ch3_feature_evolution_pure.png")