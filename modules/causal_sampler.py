import numpy as np
import random


class CausalHardNegativeSampler:
    def __init__(self, nodes, engine):
        self.engine = engine  # StreamingGraphEngine
        self.all_nodes = list(nodes)

    """
    采样逻辑：寻找和目标节点 Coreness 相似，但不是真实交互对象的节点
    size：要采多少个负样本
    current_dst_nodes：当前这一批正样本里的真实目标节点
    读取当前正样本目标节点的结构等级，从活跃节点里找结构等级相近的候选，把它作为难负样本返回
    """
    def sample(self, size, current_dst_nodes):

        # 获取当前正样本的 Coreness
        dst_cores, _ = self.engine.get_structure_features(current_dst_nodes)

        neg_samples = []
        # 简单起见，从当前活跃节点中筛选
        active_nodes = list(self.engine.node_coreness.keys())
        if not active_nodes:
            active_nodes = self.all_nodes

        """
        随机抽候选节点
        看它的结构分数 cand_core
        如果和真实目标节点 target_core 足够接近，就接受
        """
        for i in range(size):
            target_core = dst_cores[i]
            # 尝试找 5 次，找不到就随机退化
            found = False
            for _ in range(5):
                cand = random.choice(active_nodes)
                cand_core = self.engine.node_coreness.get(cand, 0)
                # 核心条件：Coreness 差距小 (相似的结构地位)
                if abs(cand_core - target_core) <= 2:
                    neg_samples.append(cand)
                    found = True
                    break

            if not found:
                neg_samples.append(random.choice(self.all_nodes))

        return np.array(neg_samples)

    def reset_random_state(self):
        pass