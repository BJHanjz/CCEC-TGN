import math
import logging
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
import csv
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

# --- 基础工具导入 ---
from utils.data_processing import get_data
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from modules.ctgn import CTGN

# --- 学位论文模块导入 ---
try:
    from modules.streaming_graph_engine import StreamingGraphEngine  # 时间衰减结构引擎
    from modules.structure_encoder import StructureEncoder  # 时间衰减结构引擎
    from modules.causal_sampler import CausalHardNegativeSampler  # 结构困难负采样器
except ImportError as e:
    print(f"Error: 缺少必要的模块。\n详情: {e}")
    sys.exit(1)

# --- 参数配置 ---
# 基础训练参数:--data  --bs  --n_epoch  --lr  --gpu  --patience
# 基础训练参数:--n_degree  --n_head  --n_layer  --node_dim  --time_dim  --use_memory
# 论文增强参数:--window_size  --lamda  --smooth_beta  --decay_factor
parser = argparse.ArgumentParser('Thesis Final V11.0: Time-Decay Structural Engine')
parser.add_argument('-d', '--data', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Early stop patience')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--window_size', type=int, default=3 * 24 * 3600, help='Window size (sec)')
parser.add_argument('--lamda', type=float, default=0.01, help='Regularization weight (0=Baseline)')
parser.add_argument('--use_memory', action='store_true', default=True, help='Whether to use memory')
parser.add_argument('--smooth_beta', type=float, default=0.2, help='Structural smoothing factor')

# === 创新点新增参数 ===
parser.add_argument('--decay_factor', type=float, default=1e-6,
                    help='Time decay factor for structural score. Suggest: 1e-6 for Wiki/Reddit')

# ==== 补充实验1参数 ====
# --ablation full  完整论文模型
# --ablation wostruct  纯 TGN baseline
# --ablation wocausal  有结构增强，但没有因果正则
# --ablation wointerp  去掉条件结构插值/平滑
# --ablation wogate  用简单加法替代门控残差
# --ablation wotimedecay  结构引擎不做时间衰减

parser.add_argument(
    '--ablation',
    type=str,
    default='full',
    choices=['full', 'wostruct', 'wocausal', 'wointerp', 'wogate', 'wotimedecay'],
    help='Ablation setting'
)
# ==== 补充实验1参数 ====


try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])

# --- 日志配置 ---
Path("log/").mkdir(parents=True, exist_ok=True)
Path("saved_models/").mkdir(parents=True, exist_ok=True)
Path("results/").mkdir(parents=True, exist_ok=True)

# 把实验模式和超参数编码进日志名里，方便后期对比不同实验结果
# ==== 补充实验1 ====
# mode_str = "BASELINE" if args.lamda == 0 else f"V11_DECAY{args.decay_factor}_BETA{args.smooth_beta}"
# ==== 补充实验1 ====


# ==== 补充实验1 ====
# if args.lamda == 0:
#     mode_str = "BASELINE" if args.ablation == 'full' else f"ABLATION_{args.ablation.upper()}"
# else:
#     effective_decay = 0.0 if args.ablation == 'wotimedecay' else args.decay_factor
#     mode_str = f"{args.ablation.upper()}_DECAY{effective_decay}_BETA{args.smooth_beta}"
# ==== 补充实验1 ====

# ==== 补充实验1 ====
if args.lamda == 0:
    mode_str = "BASELINE"
else:
    effective_decay = 0.0 if args.ablation == 'wotimedecay' else args.decay_factor
    if args.ablation == "full":
        mode_str = f"FULL_DECAY{effective_decay}_BETA{args.smooth_beta}"
    else:
        mode_str = f"{args.ablation.upper()}_DECAY{effective_decay}_BETA{args.smooth_beta}"
# ==== 补充实验1 ====

current_time_str = time.strftime("%Y%m%d_%H%M%S")

log_filename = f"log/{args.data}_bs{args.bs}_lr{args.lr}_{mode_str}_{current_time_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()
logger.info(f"Log file created at: {log_filename}")
logger.info(f"Running Mode: {mode_str}")


# --- 辅助函数 ---
# 结构增强的附加正则项的核心损失
def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)


"""
评估主逻辑:
1.把模型切到 eval()
2.按 batch 遍历验证集或测试集
4.用 engine.update_batch(src, dst, ts) 更新结构引擎
4.用负采样器采样负边
5.如果不是 baseline，就取当前 batch 的结构分数，构建 current_partition
6.调用模型算 pos_prob 和 neg_prob
7.计算 AP 和 AUC
"""


def eval_edge_prediction_thesis(model, negative_edge_sampler, data, n_neighbors, engine, batch_size=200,
                                is_inductive=False, is_baseline=False):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()
    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_instance, start_idx + batch_size)
            src = data.sources[start_idx:end_idx]
            dst = data.destinations[start_idx:end_idx]
            ts = data.timestamps[start_idx:end_idx]
            edge_idxs = data.edge_idxs[start_idx:end_idx]

            engine.update_batch(src, dst, ts)
            size = len(src)
            _, neg = negative_edge_sampler.sample(size)

            # Baseline 模式不计算结构特征
            current_partition = None
            if not is_baseline:
                all_nodes = np.concatenate([src, dst, neg])
                all_cores, _ = engine.get_structure_features(all_nodes)
                # Inductive 节点由于没有历史，分数可能为0，稍微给个底数10防止完全冷启动
                if is_inductive:
                    all_cores[all_cores == 0] = 5
                current_partition = {str(n): int(c) for n, c in zip(all_nodes, all_cores)}

            pos_prob, neg_prob = model(src, dst, neg, ts, edge_idxs, n_neighbors, partition=current_partition)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_ap), np.mean(val_auc)


# --- Baseline TGN (纯净版) ---
# 继承的是 CTGN，但把所有社群/结构增强关掉，表现成纯 TGN。
class BaselineTGN(CTGN):
    def __init__(self, num_core_levels, **kwargs):
        super().__init__(num_communities=0, **kwargs)  # 不真正使用社群 embedding

    def get_community_embeddings(self, nodes, partition):
        return None

    # 无论外部传什么 partition，都强制忽略结构增强
    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20,
                partition=None):
        return super().forward(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors,
                               partition=None)

    def forward_counterfactual(self, neg_hard_cores):
        return None, None


# --- Thesis V11 (含 Time-Decay Engine) ---
class ThesisCTGN(CTGN):
    def __init__(self, num_core_levels, **kwargs):
        self.smooth_beta = kwargs.pop('smooth_beta', 0.2)
        self.memory_updater_type = kwargs.get('memory_updater_type', 'gru')
        embed_dim = kwargs.get('memory_dimension', 100)
        dropout = kwargs.get('dropout', 0.1)
        # ==== 补充实验1 ====
        # 新增：消融开关
        self.use_structure = kwargs.pop('use_structure', True)
        self.use_causal = kwargs.pop('use_causal', True)
        self.use_interpolation = kwargs.pop('use_interpolation', True)
        self.use_gate = kwargs.pop('use_gate', True)
        # ==== 补充实验1 ====
        super().__init__(num_communities=1, **kwargs)

        self.structure_encoder = StructureEncoder(embed_dim=embed_dim, max_core=num_core_levels)
        self.structure_encoder.to(self.device)
        self.community_embedding_layer = None

        self.structure_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        ).to(self.device)

        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.latest_partition = None

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20,
                partition=None):
        self.latest_partition = partition
        # 直接调用 compute_temporal_embeddings (已重写)
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors, partition
        )
        pos_score = self.affinity_score(src_emb, dst_emb).squeeze(dim=-1)
        neg_score = self.affinity_score(src_emb, neg_emb).squeeze(dim=-1)
        return pos_score, neg_score

    """
    如果 partition 是字典，就把节点映射到对应 core 值
    如果 partition 是 tensor，也直接喂进去
    再通过 structure_encoder 编成向量
    """

    def get_community_embeddings(self, nodes, partition):
        if isinstance(partition, dict):
            core_list = [partition.get(str(n), 0) for n in nodes]
            core_tensor = torch.tensor(core_list, device=self.device)
            return self.structure_encoder(core_tensor)
        elif isinstance(partition, torch.Tensor):
            return self.structure_encoder(partition.to(self.device))
        else:
            return None

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20, partition=None):
        # 1.先拿纯 TGN 的 base embeddings
        if partition is None:
            partition = getattr(self, 'latest_partition', None)

        src_base, dst_base, neg_base = super().compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors
        )
        self.last_base_embeddings = (src_base, dst_base)

        # ==== 补充实验1 ====
        # 1) w/o Structure
        if (not self.use_structure) or partition is None:
            self.last_fused_embeddings = (src_base, dst_base)
            return src_base, dst_base, neg_base
        # ==== 补充实验1 ====

        # 2.取结构向量
        struct_src = self.get_community_embeddings(source_nodes, partition)
        struct_dst = self.get_community_embeddings(destination_nodes, partition)
        struct_neg = self.get_community_embeddings(negative_nodes, partition)

        # ==== 补充实验1 ====
        # 3.做结构平滑
        # beta = self.smooth_beta
        # smooth_src = (1.0 - beta) * struct_src + beta * struct_dst
        # smooth_dst = (1.0 - beta) * struct_dst + beta * struct_src
        # smooth_neg = (1.0 - beta) * struct_neg + beta * struct_src
        # ==== 补充实验1 ====

        # ==== 补充实验1 ====
        # 2) w/o Interpolation：不做结构平滑，直接用自身结构特征
        if self.use_interpolation:
            beta = self.smooth_beta
            smooth_src = (1.0 - beta) * struct_src + beta * struct_dst
            smooth_dst = (1.0 - beta) * struct_dst + beta * struct_src
            smooth_neg = (1.0 - beta) * struct_neg + beta * struct_src
        else:
            smooth_src = struct_src
            smooth_dst = struct_dst
            smooth_neg = struct_neg
        # ==== 补充实验1 ====

        # 4.投影:把平滑后的结构向量映射到更稳定的融合空间
        proj_struct_src = self.structure_proj(smooth_src)
        proj_struct_dst = self.structure_proj(smooth_dst)
        proj_struct_neg = self.structure_proj(smooth_neg)

        # ==== 补充实验1 ====
        # 5.门控残差融合
        # def gated_residual(tgn_emb, proj_struct_emb):
        #     concat_feat = torch.cat([tgn_emb, proj_struct_emb], dim=1)
        #     gate = self.gate_net(concat_feat)
        #     return tgn_emb + gate * proj_struct_emb
        # ==== 补充实验1 ====

        # ==== 补充实验1 ====
        # 3) w/o Gated Residual：改成简单相加
        def fuse(tgn_emb, proj_struct_emb):
            if self.use_gate:
                concat_feat = torch.cat([tgn_emb, proj_struct_emb], dim=1)
                gate = self.gate_net(concat_feat)
                return tgn_emb + gate * proj_struct_emb
            else:
                return tgn_emb + proj_struct_emb

        # ==== 补充实验1 ====

        # ==== 补充实验1 ====
        # source_embedding = gated_residual(src_base, proj_struct_src)
        # destination_embedding = gated_residual(dst_base, proj_struct_dst)
        # negative_embedding = gated_residual(neg_base, proj_struct_neg)
        # ==== 补充实验1 ====

        # ==== 补充实验1 ====
        source_embedding = fuse(src_base, proj_struct_src)
        destination_embedding = fuse(dst_base, proj_struct_dst)
        negative_embedding = fuse(neg_base, proj_struct_neg)
        # ==== 补充实验1 ====

        self.last_fused_embeddings = (source_embedding, destination_embedding)
        return source_embedding, destination_embedding, negative_embedding

    # ==== 补充实验1 ====
    # 结构反事实分支,这个函数只有在加正则时才会用到。
    # def forward_counterfactual(self, neg_hard_cores):
    #     if self.last_base_embeddings is None:
    #         raise RuntimeError("Must run normal forward first!")
    #     src_base, dst_base = self.last_base_embeddings
    #     cf_structure_emb = self.structure_encoder(neg_hard_cores.to(self.device))
    #     proj_cf_struct = self.structure_proj(cf_structure_emb)
    #     concat_feat = torch.cat([dst_base, proj_cf_struct], dim=1)
    #     gate = self.gate_net(concat_feat)
    #     cf_fused_dst = dst_base + gate * proj_cf_struct
    #     struct_importance = self.importance_net(concat_feat)
    #     return cf_fused_dst, struct_importance
    # ==== 补充实验1 ====

    # ==== 补充实验1 ====
    # 为了兼容 wostruct 或 wogate
    def forward_counterfactual(self, neg_hard_cores):
        if self.last_base_embeddings is None:
            raise RuntimeError("Must run normal forward first!")

        src_base, dst_base = self.last_base_embeddings

        if not self.use_structure:
            return dst_base, torch.ones((dst_base.size(0), 1), device=self.device)

        cf_structure_emb = self.structure_encoder(neg_hard_cores.to(self.device))
        proj_cf_struct = self.structure_proj(cf_structure_emb)

        if self.use_gate:
            concat_feat = torch.cat([dst_base, proj_cf_struct], dim=1)
            gate = self.gate_net(concat_feat)
            cf_fused_dst = dst_base + gate * proj_cf_struct
            struct_importance = self.importance_net(concat_feat)
        else:
            cf_fused_dst = dst_base + proj_cf_struct
            concat_feat = torch.cat([dst_base, proj_cf_struct], dim=1)
            struct_importance = self.importance_net(concat_feat)

        return cf_fused_dst, struct_importance
    # ==== 补充实验1 ====


# --- 主程序 ---
def main():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Parameters: {args}")

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        args.data)

    data_feat_dim = node_features.shape[1]
    if args.node_dim != data_feat_dim:  # 如果 args.node_dim 和数据本身特征维度不一致，就自动改成一致。
        logger.warning(f"Auto-correcting args.node_dim to {data_feat_dim}...")
        args.node_dim = data_feat_dim
        args.time_dim = data_feat_dim

    # ==== 补充实验1 ====
    # === 关键修改：初始化引擎时传入 Time-Decay 参数 ===
    # engine = StreamingGraphEngine(window_size_seconds=args.window_size,
    #                               time_decay_factor=args.decay_factor)
    # ==== 补充实验1 ====

    # ==== 补充实验1 ====
    # 控制结构引擎是否关掉时间衰减
    # 把所有初始化 StreamingGraphEngine 的地方，改成如下（warmup_engine() 里也同样改。）：********************************
    effective_decay = 0.0 if args.ablation == 'wotimedecay' else args.decay_factor

    engine = StreamingGraphEngine(
        window_size_seconds=args.window_size,
        time_decay_factor=effective_decay
    )
    # ==== 补充实验1 ====

    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)  # 普通随机负采样
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)  # 普通随机负采样
    causal_sampler = CausalHardNegativeSampler(full_data.unique_nodes, engine)  # 普通随机负采样

    # 初始化邻居查找器
    train_ngh_finder = get_neighbor_finder(train_data, uniform=False)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False)

    """
    如果 lamda == 0,构造 BaselineTGN，也就是纯 TGN 模式。
    否则,构造 ThesisCTGN，启用：结构编码、平滑、门控残差增强、反事实结构正则
    """
    # ==== 补充实验1 ====
    # is_baseline_mode = (args.lamda == 0)
    #
    # if is_baseline_mode:
    #     logger.info(">>> Running BASELINE TGN (Pure Mode) <<<")
    #     model = BaselineTGN(num_core_levels=0, neighbor_finder=train_ngh_finder,
    #                         node_features=node_features, edge_features=edge_features, device=device,
    #                         n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out,
    #                         use_memory=args.use_memory, memory_dimension=args.node_dim,
    #                         embedding_module_type="graph_attention", message_function="identity",
    #                         aggregator_type="last", memory_updater_type="gru"
    #                         ).to(device)
    # else:
    #     logger.info(f">>> Running THESIS MODEL (Smoothing V11 | Decay={args.decay_factor}) <<<")
    #     model = ThesisCTGN(
    #         num_core_levels=200,  # 依然使用 200 个分级，引擎会自动映射
    #         neighbor_finder=train_ngh_finder,
    #         node_features=node_features, edge_features=edge_features, device=device,
    #         n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out,
    #         smooth_beta=args.smooth_beta,
    #         use_memory=args.use_memory, memory_dimension=args.node_dim,
    #         embedding_module_type="graph_attention", message_function="identity",
    #         aggregator_type="last", memory_updater_type="gru"
    #     ).to(device)
    # ==== 补充实验1 ====

    # ==== 补充实验1 ====
    # 在 main() 里根据 args.ablation 组装模型，把模型初始化部分改成：
    is_baseline_mode = (args.lamda == 0)

    if is_baseline_mode:
        logger.info(">>> Running BASELINE MODE (lamda=0) <<<")
        model = BaselineTGN(
            num_core_levels=0,
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=args.n_layer,
            n_heads=args.n_head,
            dropout=args.drop_out,
            use_memory=args.use_memory,
            memory_dimension=args.node_dim,
            embedding_module_type="graph_attention",
            message_function="identity",
            aggregator_type="last",
            memory_updater_type="gru"
        ).to(device)
    else:
        logger.info(f">>> Running THESIS MODEL | ablation={args.ablation} <<<")

        use_structure = (args.ablation != 'wostruct')
        use_causal = (args.ablation != 'wocausal')
        use_interpolation = (args.ablation != 'wointerp')
        use_gate = (args.ablation != 'wogate')

        model = ThesisCTGN(
            num_core_levels=200,
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=args.n_layer,
            n_heads=args.n_head,
            dropout=args.drop_out,
            smooth_beta=args.smooth_beta,
            use_memory=args.use_memory,
            memory_dimension=args.node_dim,
            embedding_module_type="graph_attention",
            message_function="identity",
            aggregator_type="last",
            memory_updater_type="gru",
            use_structure=use_structure,
            use_causal=use_causal,
            use_interpolation=use_interpolation,
            use_gate=use_gate
        ).to(device)
    # ==== 补充实验1 ====

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopMonitor(max_round=args.patience)

    # ==== 补充实验1 ====
    if args.lamda == 0:
        save_suffix = "baseline"
    elif args.ablation in ["wostruct", "wocausal", "wointerp", "wogate", "wotimedecay"]:
        save_suffix = args.ablation
    else:
        save_suffix = "full"
    model_save_path = f'saved_models/thesis_final_{args.data}_{save_suffix}.pth'
    # ==== 补充实验1 ====

    # ==== 补充实验1 ====
    #model_save_path = f'saved_models/thesis_final_{args.data}.pth'
    # ==== 补充实验1 ====

    num_batch = math.ceil(len(train_data.sources) / args.bs)
    logger.info(f"Start training: {args.n_epoch} epochs")

    for epoch in range(args.n_epoch):
        start_time = time.time()
        model.train()
        model.set_neighbor_finder(train_ngh_finder)
        if args.use_memory: model.memory.__init_memory__()

        # ==== 补充实验1 ====
        # 每个 epoch 重新初始化引擎
        # engine = StreamingGraphEngine(window_size_seconds=args.window_size,
        #                               time_decay_factor=args.decay_factor)
        # ==== 补充实验1 ====

        # ==== 补充实验1 ====
        # 控制结构引擎是否关掉时间衰减
        # 把所有初始化 StreamingGraphEngine 的地方，改成如下（warmup_engine() 里也同样改。）：********************************
        effective_decay = 0.0 if args.ablation == 'wotimedecay' else args.decay_factor

        engine = StreamingGraphEngine(
            window_size_seconds=args.window_size,
            time_decay_factor=effective_decay
        )
        # ==== 补充实验1 ====

        batch_losses = []
        for i in range(num_batch):
            start_idx = i * args.bs
            end_idx = min((i + 1) * args.bs, len(train_data.sources))
            # 取这批边
            src, dst = train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx]
            ts, e_idx = train_data.timestamps[start_idx:end_idx], train_data.edge_idxs[start_idx:end_idx]

            # 更新结构引擎
            engine.update_batch(src, dst, ts)

            size = len(src)
            _, neg_rand = train_rand_sampler.sample(size)  # 采随机负样本

            current_partition = None
            # 如果不是 baseline，构造 current_partition：当前 batch 里涉及到的节点，它们在结构引擎中的离散结构等级
            if not is_baseline_mode:
                all_nodes = np.concatenate([src, dst, neg_rand])
                all_cores, _ = engine.get_structure_features(all_nodes)
                current_partition = {str(n): int(c) for n, c in zip(all_nodes, all_cores)}

            optimizer.zero_grad()
            # 主损失
            pos_prob, neg_prob = model(src, dst, neg_rand, ts, e_idx, args.n_degree, partition=current_partition)
            loss = criterion(pos_prob, torch.ones_like(pos_prob)) + criterion(neg_prob, torch.zeros_like(neg_prob))

            """
            额外正则项：只有 thesis 模式才加
            1.用 causal_sampler 采结构困难负样本 neg_hard
            2.从结构引擎里取这些难负样本的结构等级 neg_hard_cores
            3.用 model.forward_counterfactual(...) 生成反事实目的节点表示
            4.用 info_nce_loss 约束真实融合表示和反事实表示
            5.再乘以 adaptive_weight = mean(struct_importance)
            6.加到总损失上。
            """

            # ==== 补充实验1 ====
            # if args.lamda > 0:
            # ==== 补充实验1 ====

            # ==== 补充实验1 ====
            if args.lamda > 0 and (args.ablation != 'wocausal') and (not is_baseline_mode):
            # ==== 补充实验1 ====

                neg_hard = causal_sampler.sample(size, dst)
                neg_hard_cores, _ = engine.get_structure_features(neg_hard)
                cf_dst_emb, struct_importance = model.forward_counterfactual(
                    torch.tensor(neg_hard_cores, dtype=torch.long))
                reg_loss_val = info_nce_loss(model.last_fused_embeddings[1], cf_dst_emb)
                adaptive_weight = torch.mean(struct_importance)
                loss += args.lamda * adaptive_weight * reg_loss_val

            # 反向传播与 memory 截断：detach_memory()避免计算图跨过太多 batch 无限延伸
            loss.backward()
            optimizer.step()
            if args.use_memory: model.memory.detach_memory()
            batch_losses.append(loss.item())

        # 每个 epoch 结束：做验证和早停
        # early stop 的依据是 验证集 AP，不是 loss，early stop 的依据是 验证集 AP，不是 loss
        model.eval()
        model.set_neighbor_finder(full_ngh_finder)
        val_ap, val_auc = eval_edge_prediction_thesis(model, val_rand_sampler, val_data, args.n_degree, engine, args.bs,
                                                      is_baseline=is_baseline_mode)

        logger.info(
            f"Epoch {epoch + 1}: Loss {np.mean(batch_losses):.4f}, Val AP {val_ap:.4f}, Time {time.time() - start_time:.2f}s")
        if early_stopper.early_stop_check(val_ap):
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Best model saved to {model_save_path}")
            break

    if args.n_epoch > 0 and not early_stopper.early_stop_check(val_ap):
        torch.save(model.state_dict(), model_save_path)

    logger.info(">>> Starting Final Testing Phase <<<")

    # ==== 补充实验1 ====
    # def warmup_engine(data_list):
    #     new_engine = StreamingGraphEngine(window_size_seconds=args.window_size,
    #                                       time_decay_factor=args.decay_factor)
    #     for d in data_list: new_engine.update_batch(d.sources, d.destinations, d.timestamps)
    #     return new_engine
    # ==== 补充实验1 ====
    def warmup_engine(data_list):
        effective_decay = 0.0 if args.ablation == 'wotimedecay' else args.decay_factor

        new_engine = StreamingGraphEngine(
            window_size_seconds=args.window_size,
            time_decay_factor=effective_decay
        )
        for d in data_list:
            new_engine.update_batch(d.sources, d.destinations, d.timestamps)
        return new_engine

    # Transductive Test：先用 warmup_engine([train_data, val_data])
    # 让结构引擎“看到”训练+验证阶段的数据流，然后在 test_data 上评估
    if os.path.exists(model_save_path): model.load_state_dict(torch.load(model_save_path))
    model.eval()
    model.set_neighbor_finder(full_ngh_finder)
    test_engine = warmup_engine([train_data, val_data])
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    test_ap, test_auc = eval_edge_prediction_thesis(model, test_rand_sampler, test_data, args.n_degree, test_engine,
                                                    args.bs, is_inductive=False, is_baseline=is_baseline_mode)

    # Inductive Test：再重置 memory，并用 new_node_test_data 做新节点测试。
    # 这里用的是专门的新节点随机采样器和 inductive 标志
    if args.use_memory: model.memory.__init_memory__()
    model.eval()
    model.set_neighbor_finder(full_ngh_finder)
    inductive_engine = warmup_engine([train_data, val_data])
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=42)
    nn_test_ap, nn_test_auc = eval_edge_prediction_thesis(model, nn_test_rand_sampler, new_node_test_data,
                                                          args.n_degree, inductive_engine, args.bs, is_inductive=True,
                                                          is_baseline=is_baseline_mode)

    logger.info("-" * 50)
    logger.info(f"FINAL RESULTS ({args.data} | Mode: {mode_str}):")
    # 【修改点1】同时打印 AP 和 AUC
    logger.info(f"Transductive Test AP: {test_ap:.4f} | AUC: {test_auc:.4f}")
    logger.info(f"Inductive Test AP:    {nn_test_ap:.4f} | AUC: {nn_test_auc:.4f}")
    logger.info("-" * 50)

    try:
        with open("results/experiment_results.csv", mode='a', newline='') as f:
            # ==== 补充实验1 ====
            # # 【修改点2】把 AUC 也写入 CSV 表格中 (注意表头要对应)
            # csv.writer(f).writerow([
            #     time.strftime("%Y-%m-%d %H:%M:%S"), args.data, mode_str, args.bs, args.lr, args.lamda,
            #     f"AP:{test_ap:.4f}", f"AUC:{test_auc:.4f}", f"In-AP:{nn_test_ap:.4f}", f"In-AUC:{nn_test_auc:.4f}"
            # ])
            # ==== 补充实验1 ====

            # ==== 补充实验1 ====
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.data,
                mode_str,
                args.ablation,
                args.bs,
                args.lr,
                args.lamda,
                f"AP:{test_ap:.4f}",
                f"AUC:{test_auc:.4f}",
                f"In-AP:{nn_test_ap:.4f}",
                f"In-AUC:{nn_test_auc:.4f}"
            ])
            # ==== 补充实验1 ====

            # ==== 补充实验2 ====
            # csv.writer(f).writerow([
            #     time.strftime("%Y-%m-%d %H:%M:%S"),
            #     args.exp_type,
            #     args.data,
            #     mode_str,
            #     args.ablation,
            #     args.bs,
            #     args.lr,
            #     args.decay_factor,
            #     args.smooth_beta,
            #     args.lamda,
            #     f"{test_ap:.4f}",
            #     f"{test_auc:.4f}",
            #     f"{nn_test_ap:.4f}",
            #     f"{nn_test_auc:.4f}"
            # ])
            # ==== 补充实验2 ====

    except Exception as e:
        logger.error(f"Save CSV failed: {e}")


if __name__ == "__main__":
    main()
#     logger.info("-" * 50)
#     logger.info(f"FINAL RESULTS ({args.data} | Mode: {mode_str}):")
#     logger.info(f"Transductive Test AP: {test_ap:.4f}")
#     logger.info(f"Inductive Test AP:    {nn_test_ap:.4f}")
#     logger.info("-" * 50)
#
#     try:
#         with open("results/experiment_results.csv", mode='a', newline='') as f:
#             csv.writer(f).writerow([
#                 time.strftime("%Y-%m-%d %H:%M:%S"), args.data, mode_str, args.bs, args.lr, args.lamda,
#                 f"{test_ap:.4f}", f"{nn_test_ap:.4f}"
#             ])
#     except:
#         pass
#
#
# if __name__ == "__main__":
#     main()
