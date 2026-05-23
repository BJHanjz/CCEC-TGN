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
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

# --- 基础工具导入 ---
from utils.data_processing import get_data
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from modules.ctgn import CTGN

# --- 学位论文模块导入 ---
try:
    from modules.streaming_graph_engine import StreamingGraphEngine
    from modules.structure_encoder import StructureEncoder
    from modules.causal_sampler import CausalHardNegativeSampler
except ImportError as e:
    print(f"Error: 缺少必要的模块。\n详情: {e}")
    sys.exit(1)


# =========================================================
# 路径与全局配置
# =========================================================
Path("log/").mkdir(parents=True, exist_ok=True)
Path("saved_models/").mkdir(parents=True, exist_ok=True)
Path("results/").mkdir(parents=True, exist_ok=True)

RAW_CSV = "results/sensitivity_exp2_raw.csv"
SUMMARY_CSV = "results/sensitivity_exp2_summary.csv"

DECAY_LIST = [0.0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
BETA_LIST = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
LAMDA_LIST = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]


# =========================================================
# 参数配置
# =========================================================
parser = argparse.ArgumentParser("Experiment-2 Hyperparameter Sensitivity All-in-One")

# 运行模式
parser.add_argument(
    "--mode",
    type=str,
    default="batch",
    choices=["single", "batch"],
    help="single: 跑单次实验; batch: 一键批量跑实验二"
)

# 批处理设置
parser.add_argument(
    "--target",
    type=str,
    default="all",
    choices=["all", "decay", "beta", "lamda"],
    help="batch 模式下跑哪一组敏感性实验"
)
parser.add_argument(
    "--repeat",
    type=int,
    default=1,
    help="batch 模式下每个参数值重复次数"
)

# 单次实验标识
parser.add_argument(
    "--exp_type",
    type=str,
    default="main",
    choices=["main", "sensitivity_decay", "sensitivity_beta", "sensitivity_lamda"],
    help="single 模式下的实验类型"
)
parser.add_argument(
    "--sweep_param",
    type=str,
    default="none",
    choices=["none", "decay_factor", "smooth_beta", "lamda"],
    help="single 模式下当前扫描的参数"
)
parser.add_argument(
    "--sweep_value",
    type=float,
    default=-1.0,
    help="single 模式下当前参数值"
)
parser.add_argument(
    "--run_id",
    type=int,
    default=1,
    help="single 模式下重复实验编号"
)

# 基础训练参数
parser.add_argument('-d', '--data', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--bs', type=int, default=200, help='Batch size')
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
parser.add_argument('--lamda', type=float, default=0.01, help='Regularization weight (0=Baseline in original design)')
parser.add_argument('--use_memory', action='store_true', default=True, help='Whether to use memory')
parser.add_argument('--smooth_beta', type=float, default=0.2, help='Structural smoothing factor')
parser.add_argument('--decay_factor', type=float, default=1e-6, help='Time decay factor for structural score')

parser.add_argument(
    '--ablation',
    type=str,
    default='full',
    choices=['full', 'wostruct', 'wocausal', 'wointerp', 'wogate', 'wotimedecay'],
    help='Ablation setting'
)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])


# =========================================================
# 日志
# =========================================================
def build_logger(local_args):
    if local_args.lamda == 0:
        mode_str = "BASELINE"
    else:
        effective_decay = 0.0 if local_args.ablation == 'wotimedecay' else local_args.decay_factor
        if local_args.ablation == 'full':
            mode_str = f"FULL_DECAY{effective_decay}_BETA{local_args.smooth_beta}"
        else:
            mode_str = f"{local_args.ablation.upper()}_DECAY{effective_decay}_BETA{local_args.smooth_beta}"

    current_time_str = time.strftime("%Y%m%d_%H%M%S")

    log_filename = (
        f"log/{local_args.data}_"
        f"{local_args.exp_type}_"
        f"ablation{local_args.ablation}_"
        f"{local_args.sweep_param}{local_args.sweep_value}_"
        f"run{local_args.run_id}_"
        f"bs{local_args.bs}_lr{local_args.lr}_"
        f"decay{local_args.decay_factor}_beta{local_args.smooth_beta}_lamda{local_args.lamda}_"
        f"{current_time_str}.log"
    )

    logger_name = f"sensitivity_exp2_{time.time_ns()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Log file created at: {log_filename}")
    logger.info(f"Running mode: {mode_str}")
    logger.info(f"Experiment Type: {local_args.exp_type}")
    logger.info(f"Sweep Param: {local_args.sweep_param}")
    logger.info(f"Sweep Value: {local_args.sweep_value}")
    logger.info(f"Run ID: {local_args.run_id}")

    return logger, log_filename, mode_str


# =========================================================
# 辅助函数
# =========================================================
def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)


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

            current_partition = None
            if not is_baseline:
                all_nodes = np.concatenate([src, dst, neg])
                all_cores, _ = engine.get_structure_features(all_nodes)
                if is_inductive:
                    all_cores[all_cores == 0] = 5
                current_partition = {str(n): int(c) for n, c in zip(all_nodes, all_cores)}

            pos_prob, neg_prob = model(src, dst, neg, ts, edge_idxs, n_neighbors, partition=current_partition)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


# =========================================================
# Baseline TGN
# =========================================================
class BaselineTGN(CTGN):
    def __init__(self, num_core_levels, **kwargs):
        super().__init__(num_communities=0, **kwargs)

    def get_community_embeddings(self, nodes, partition):
        return None

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20,
                partition=None):
        return super().forward(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors,
                               partition=None)

    def forward_counterfactual(self, neg_hard_cores):
        return None, None


# =========================================================
# Thesis V11
# =========================================================
class ThesisCTGN(CTGN):
    def __init__(self, num_core_levels, **kwargs):
        self.smooth_beta = kwargs.pop('smooth_beta', 0.2)
        self.memory_updater_type = kwargs.get('memory_updater_type', 'gru')
        embed_dim = kwargs.get('memory_dimension', 100)
        dropout = kwargs.get('dropout', 0.1)

        self.use_structure = kwargs.pop('use_structure', True)
        self.use_causal = kwargs.pop('use_causal', True)
        self.use_interpolation = kwargs.pop('use_interpolation', True)
        self.use_gate = kwargs.pop('use_gate', True)

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
        self.last_base_embeddings = None
        self.last_fused_embeddings = None

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20,
                partition=None):
        self.latest_partition = partition
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors, partition
        )
        pos_score = self.affinity_score(src_emb, dst_emb).squeeze(dim=-1)
        neg_score = self.affinity_score(src_emb, neg_emb).squeeze(dim=-1)
        return pos_score, neg_score

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
        if partition is None:
            partition = getattr(self, 'latest_partition', None)

        src_base, dst_base, neg_base = super().compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors
        )
        self.last_base_embeddings = (src_base, dst_base)

        if (not self.use_structure) or partition is None:
            self.last_fused_embeddings = (src_base, dst_base)
            return src_base, dst_base, neg_base

        struct_src = self.get_community_embeddings(source_nodes, partition)
        struct_dst = self.get_community_embeddings(destination_nodes, partition)
        struct_neg = self.get_community_embeddings(negative_nodes, partition)

        if self.use_interpolation:
            beta = self.smooth_beta
            smooth_src = (1.0 - beta) * struct_src + beta * struct_dst
            smooth_dst = (1.0 - beta) * struct_dst + beta * struct_src
            smooth_neg = (1.0 - beta) * struct_neg + beta * struct_src
        else:
            smooth_src = struct_src
            smooth_dst = struct_dst
            smooth_neg = struct_neg

        proj_struct_src = self.structure_proj(smooth_src)
        proj_struct_dst = self.structure_proj(smooth_dst)
        proj_struct_neg = self.structure_proj(smooth_neg)

        def fuse(tgn_emb, proj_struct_emb):
            if self.use_gate:
                concat_feat = torch.cat([tgn_emb, proj_struct_emb], dim=1)
                gate = self.gate_net(concat_feat)
                return tgn_emb + gate * proj_struct_emb
            else:
                return tgn_emb + proj_struct_emb

        source_embedding = fuse(src_base, proj_struct_src)
        destination_embedding = fuse(dst_base, proj_struct_dst)
        negative_embedding = fuse(neg_base, proj_struct_neg)

        self.last_fused_embeddings = (source_embedding, destination_embedding)
        return source_embedding, destination_embedding, negative_embedding

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


# =========================================================
# 单次训练
# =========================================================
def run_single_experiment(local_args):
    logger, log_filename, mode_str = build_logger(local_args)

    device = torch.device(f'cuda:{local_args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Parameters: {local_args}")

    random.seed(local_args.run_id)
    np.random.seed(local_args.run_id)
    torch.manual_seed(local_args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_args.run_id)

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        local_args.data)

    data_feat_dim = node_features.shape[1]
    if local_args.node_dim != data_feat_dim:
        logger.warning(f"Auto-correcting args.node_dim to {data_feat_dim}...")
        local_args.node_dim = data_feat_dim
        local_args.time_dim = data_feat_dim

    effective_decay = 0.0 if local_args.ablation == 'wotimedecay' else local_args.decay_factor

    engine = StreamingGraphEngine(
        window_size_seconds=local_args.window_size,
        time_decay_factor=effective_decay
    )

    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    causal_sampler = CausalHardNegativeSampler(full_data.unique_nodes, engine)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False)

    # 保持与原始脚本一致：lamda=0 进入 baseline
    is_baseline_mode = (local_args.lamda == 0)

    if is_baseline_mode:
        logger.info(">>> Running BASELINE MODE (lamda=0) <<<")
        model = BaselineTGN(
            num_core_levels=0,
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=local_args.n_layer,
            n_heads=local_args.n_head,
            dropout=local_args.drop_out,
            use_memory=local_args.use_memory,
            memory_dimension=local_args.node_dim,
            embedding_module_type="graph_attention",
            message_function="identity",
            aggregator_type="last",
            memory_updater_type="gru"
        ).to(device)
    else:
        logger.info(f">>> Running THESIS MODEL | ablation={local_args.ablation} <<<")

        use_structure = (local_args.ablation != 'wostruct')
        use_causal = (local_args.ablation != 'wocausal')
        use_interpolation = (local_args.ablation != 'wointerp')
        use_gate = (local_args.ablation != 'wogate')

        model = ThesisCTGN(
            num_core_levels=200,
            neighbor_finder=train_ngh_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=local_args.n_layer,
            n_heads=local_args.n_head,
            dropout=local_args.drop_out,
            smooth_beta=local_args.smooth_beta,
            use_memory=local_args.use_memory,
            memory_dimension=local_args.node_dim,
            embedding_module_type="graph_attention",
            message_function="identity",
            aggregator_type="last",
            memory_updater_type="gru",
            use_structure=use_structure,
            use_causal=use_causal,
            use_interpolation=use_interpolation,
            use_gate=use_gate
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=local_args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopMonitor(max_round=local_args.patience)
    if local_args.lamda == 0:
        save_suffix = "baseline"
    elif local_args.ablation in ["wostruct", "wocausal", "wointerp", "wogate", "wotimedecay"]:
        save_suffix = local_args.ablation
    else:
        save_suffix = "full"

    model_save_path = (
        f"saved_models/sensitivity_exp2_{local_args.data}_"
        f"{local_args.exp_type}_{local_args.sweep_param}_{local_args.sweep_value}_"
        f"{save_suffix}_run{local_args.run_id}.pth"
    )

    num_batch = math.ceil(len(train_data.sources) / local_args.bs)
    logger.info(f"Start training: {local_args.n_epoch} epochs")

    for epoch in range(local_args.n_epoch):
        start_time = time.time()
        model.train()
        model.set_neighbor_finder(train_ngh_finder)
        if local_args.use_memory:
            model.memory.__init_memory__()

        engine = StreamingGraphEngine(
            window_size_seconds=local_args.window_size,
            time_decay_factor=local_args.decay_factor
        )

        batch_losses = []
        for i in range(num_batch):
            start_idx = i * local_args.bs
            end_idx = min((i + 1) * local_args.bs, len(train_data.sources))

            src = train_data.sources[start_idx:end_idx]
            dst = train_data.destinations[start_idx:end_idx]
            ts = train_data.timestamps[start_idx:end_idx]
            e_idx = train_data.edge_idxs[start_idx:end_idx]

            engine.update_batch(src, dst, ts)

            size = len(src)
            _, neg_rand = train_rand_sampler.sample(size)

            current_partition = None
            if not is_baseline_mode:
                all_nodes = np.concatenate([src, dst, neg_rand])
                all_cores, _ = engine.get_structure_features(all_nodes)
                current_partition = {str(n): int(c) for n, c in zip(all_nodes, all_cores)}

            optimizer.zero_grad()

            pos_prob, neg_prob = model(src, dst, neg_rand, ts, e_idx, local_args.n_degree, partition=current_partition)
            loss = criterion(pos_prob, torch.ones_like(pos_prob)) + criterion(neg_prob, torch.zeros_like(neg_prob))

            if local_args.lamda > 0 and (local_args.ablation != 'wocausal') and (not is_baseline_mode):
                neg_hard = causal_sampler.sample(size, dst)
                neg_hard_cores, _ = engine.get_structure_features(neg_hard)
                cf_dst_emb, struct_importance = model.forward_counterfactual(
                    torch.tensor(neg_hard_cores, dtype=torch.long)
                )
                reg_loss_val = info_nce_loss(model.last_fused_embeddings[1], cf_dst_emb)
                adaptive_weight = torch.mean(struct_importance)
                loss += local_args.lamda * adaptive_weight * reg_loss_val

            loss.backward()
            optimizer.step()

            if local_args.use_memory:
                model.memory.detach_memory()

            batch_losses.append(loss.item())

        model.eval()
        model.set_neighbor_finder(full_ngh_finder)
        val_ap, val_auc = eval_edge_prediction_thesis(
            model, val_rand_sampler, val_data, local_args.n_degree, engine, local_args.bs,
            is_baseline=is_baseline_mode
        )

        logger.info(
            f"Epoch {epoch + 1}: Loss {np.mean(batch_losses):.4f}, "
            f"Val AP {val_ap:.4f}, Val AUC {val_auc:.4f}, "
            f"Time {time.time() - start_time:.2f}s"
        )

        if early_stopper.early_stop_check(val_ap):
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Best model saved to {model_save_path}")
            break

    if local_args.n_epoch > 0 and not early_stopper.early_stop_check(val_ap):
        torch.save(model.state_dict(), model_save_path)

    logger.info(">>> Starting Final Testing Phase <<<")

    def warmup_engine(data_list):
        new_engine = StreamingGraphEngine(
            window_size_seconds=local_args.window_size,
            time_decay_factor=local_args.decay_factor
        )
        for d in data_list:
            new_engine.update_batch(d.sources, d.destinations, d.timestamps)
        return new_engine

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    model.eval()
    model.set_neighbor_finder(full_ngh_finder)

    test_engine = warmup_engine([train_data, val_data])
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    test_ap, test_auc = eval_edge_prediction_thesis(
        model, test_rand_sampler, test_data, local_args.n_degree, test_engine,
        local_args.bs, is_inductive=False, is_baseline=is_baseline_mode
    )

    if local_args.use_memory:
        model.memory.__init_memory__()

    model.eval()
    model.set_neighbor_finder(full_ngh_finder)

    inductive_engine = warmup_engine([train_data, val_data])
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=42)
    nn_test_ap, nn_test_auc = eval_edge_prediction_thesis(
        model, nn_test_rand_sampler, new_node_test_data,
        local_args.n_degree, inductive_engine, local_args.bs,
        is_inductive=True, is_baseline=is_baseline_mode
    )

    logger.info("-" * 60)
    logger.info(f"FINAL RESULTS ({local_args.data} | Mode: {mode_str})")
    logger.info(f"Transductive Test AP: {test_ap:.4f} | AUC: {test_auc:.4f}")
    logger.info(f"Inductive Test AP:    {nn_test_ap:.4f} | AUC: {nn_test_auc:.4f}")
    logger.info("-" * 60)

    result = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ablation": local_args.ablation,
        "exp_type": local_args.exp_type,
        "sweep_param": local_args.sweep_param,
        "sweep_value": local_args.sweep_value,
        "run_id": local_args.run_id,
        "data": local_args.data,
        "mode": mode_str,
        "bs": local_args.bs,
        "lr": local_args.lr,
        "decay_factor": local_args.decay_factor,
        "smooth_beta": local_args.smooth_beta,
        "lamda": local_args.lamda,
        "test_ap": test_ap,
        "test_auc": test_auc,
        "inductive_ap": nn_test_ap,
        "inductive_auc": nn_test_auc,
        "log_file": log_filename,
        "status": "success",
    }
    return result


# =========================================================
# CSV 与统计
# =========================================================
def append_csv_row(csv_path, row, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_results():
    if not os.path.exists(RAW_CSV):
        print(f"[WARN] raw csv not found: {RAW_CSV}")
        return

    df = pd.read_csv(RAW_CSV)
    df = df[df["status"] == "success"].copy()

    if df.empty:
        print("[WARN] no successful records to summarize")
        return

    grouped = df.groupby(["exp_type", "sweep_param", "sweep_value"], as_index=False).agg({
        "test_ap": ["mean", "std"],
        "test_auc": ["mean", "std"],
        "inductive_ap": ["mean", "std"],
        "inductive_auc": ["mean", "std"],
    })

    grouped.columns = [
        "exp_type", "sweep_param", "sweep_value",
        "test_ap_mean", "test_ap_std",
        "test_auc_mean", "test_auc_std",
        "inductive_ap_mean", "inductive_ap_std",
        "inductive_auc_mean", "inductive_auc_std",
    ]
    grouped = grouped.sort_values(by=["exp_type", "sweep_value"])
    grouped.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[SAVE] summary csv -> {SUMMARY_CSV}")


def plot_sensitivity():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[WARN] summary csv not found: {SUMMARY_CSV}")
        return

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        print("[WARN] summary csv is empty")
        return

    configs = [
        ("sensitivity_decay", "decay_factor", "Decay Factor"),
        ("sensitivity_beta", "smooth_beta", "Smooth Beta"),
        ("sensitivity_lamda", "lamda", "Lamda"),
    ]
    metrics = [
        ("test_ap_mean", "test_ap_std", "Transductive AP"),
        ("test_auc_mean", "test_auc_std", "Transductive AUC"),
        ("inductive_ap_mean", "inductive_ap_std", "Inductive AP"),
        ("inductive_auc_mean", "inductive_auc_std", "Inductive AUC"),
    ]

    for exp_name, sweep_name, xlabel in configs:
        sub = df[df["exp_type"] == exp_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by="sweep_value")

        for mean_col, std_col, ylabel in metrics:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sub["sweep_value"],
                sub[mean_col],
                yerr=sub[std_col].fillna(0.0),
                fmt='-o',
                capsize=4
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"{exp_name}: {ylabel}")
            plt.grid(True)
            plt.tight_layout()
            save_path = f"results/{exp_name}_{mean_col}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"[SAVE] figure -> {save_path}")


# =========================================================
# 批处理
# =========================================================
def clone_args(base_args, **updates):
    new_parser = argparse.Namespace(**vars(base_args))
    for k, v in updates.items():
        setattr(new_parser, k, v)
    return new_parser


def run_batch(base_args):
    fieldnames = [
        "time", "exp_type", "sweep_param", "sweep_value", "run_id",
        "data","ablation", "mode", "bs", "lr",
        "decay_factor", "smooth_beta", "lamda",
        "test_ap", "test_auc", "inductive_ap", "inductive_auc",
        "log_file", "status"
    ]

    if base_args.target in ["decay", "all"]:
        for value in DECAY_LIST:
            for run_id in range(1, base_args.repeat + 1):
                local_args = clone_args(
                    base_args,
                    mode="single",
                    exp_type="sensitivity_decay",
                    sweep_param="decay_factor",
                    sweep_value=value,
                    decay_factor=value,
                    smooth_beta=0.2,
                    lamda=0.1,
                    ablation="full",
                    run_id=run_id
                )
                print("=" * 100)
                print(f"[BATCH] sensitivity_decay | decay_factor={value} | run={run_id}")
                try:
                    row = run_single_experiment(local_args)
                except Exception as e:
                    row = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exp_type": local_args.exp_type,
                        "sweep_param": local_args.sweep_param,
                        "sweep_value": local_args.sweep_value,
                        "run_id": local_args.run_id,
                        "data": local_args.data,
                        "ablation": local_args.ablation,
                        "mode": "FAILED",
                        "bs": local_args.bs,
                        "lr": local_args.lr,
                        "decay_factor": local_args.decay_factor,
                        "smooth_beta": local_args.smooth_beta,
                        "lamda": local_args.lamda,
                        "test_ap": None,
                        "test_auc": None,
                        "inductive_ap": None,
                        "inductive_auc": None,
                        "log_file": "",
                        "status": f"failed: {e}",
                    }
                append_csv_row(RAW_CSV, row, fieldnames)

    if base_args.target in ["beta", "all"]:
        for value in BETA_LIST:
            for run_id in range(1, base_args.repeat + 1):
                local_args = clone_args(
                    base_args,
                    mode="single",
                    exp_type="sensitivity_beta",
                    sweep_param="smooth_beta",
                    sweep_value=value,
                    decay_factor=1e-6,
                    smooth_beta=value,
                    lamda=0.1,
                    ablation="full",
                    run_id=run_id
                )
                print("=" * 100)
                print(f"[BATCH] sensitivity_beta | smooth_beta={value} | run={run_id}")
                try:
                    row = run_single_experiment(local_args)
                except Exception as e:
                    row = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exp_type": local_args.exp_type,
                        "sweep_param": local_args.sweep_param,
                        "sweep_value": local_args.sweep_value,
                        "run_id": local_args.run_id,
                        "data": local_args.data,
                        "ablation": local_args.ablation,
                        "mode": "FAILED",
                        "bs": local_args.bs,
                        "lr": local_args.lr,
                        "decay_factor": local_args.decay_factor,
                        "smooth_beta": local_args.smooth_beta,
                        "lamda": local_args.lamda,
                        "test_ap": None,
                        "test_auc": None,
                        "inductive_ap": None,
                        "inductive_auc": None,
                        "log_file": "",
                        "status": f"failed: {e}",
                    }
                append_csv_row(RAW_CSV, row, fieldnames)

    if base_args.target in ["lamda", "all"]:
        for value in LAMDA_LIST:
            for run_id in range(1, base_args.repeat + 1):
                local_args = clone_args(
                    base_args,
                    mode="single",
                    exp_type="sensitivity_lamda",
                    sweep_param="lamda",
                    sweep_value=value,
                    decay_factor=1e-6,
                    smooth_beta=0.2,
                    lamda=value,
                    ablation="full",
                    run_id=run_id
                )
                print("=" * 100)
                print(f"[BATCH] sensitivity_lamda | lamda={value} | run={run_id}")
                try:
                    row = run_single_experiment(local_args)
                except Exception as e:
                    row = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exp_type": local_args.exp_type,
                        "sweep_param": local_args.sweep_param,
                        "sweep_value": local_args.sweep_value,
                        "run_id": local_args.run_id,
                        "data": local_args.data,
                        "mode": "FAILED",
                        "bs": local_args.bs,
                        "lr": local_args.lr,
                        "decay_factor": local_args.decay_factor,
                        "smooth_beta": local_args.smooth_beta,
                        "lamda": local_args.lamda,
                        "test_ap": None,
                        "test_auc": None,
                        "inductive_ap": None,
                        "inductive_auc": None,
                        "log_file": "",
                        "status": f"failed: {e}",
                    }
                append_csv_row(RAW_CSV, row, fieldnames)

    summarize_results()
    plot_sensitivity()
    print("[DONE] batch sensitivity finished.")


# =========================================================
# 主入口
# =========================================================
def main():
    if args.mode == "single":
        result = run_single_experiment(args)
        fieldnames = [
            "time", "exp_type", "sweep_param", "sweep_value", "run_id",
            "data", "ablation", "mode", "bs", "lr",
            "decay_factor", "smooth_beta", "lamda",
            "test_ap", "test_auc", "inductive_ap", "inductive_auc",
            "log_file", "status"
        ]
        append_csv_row(RAW_CSV, result, fieldnames)
        summarize_results()
        plot_sensitivity()
    else:
        run_batch(args)


if __name__ == "__main__":
    main()