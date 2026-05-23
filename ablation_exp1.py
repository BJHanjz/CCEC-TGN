import math
import logging
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import csv
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.data_processing import get_data
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from modules.ctgn import CTGN

try:
    from modules.streaming_graph_engine import StreamingGraphEngine
    from modules.structure_encoder import StructureEncoder
    from modules.causal_sampler import CausalHardNegativeSampler
except ImportError as e:
    print(f"Error: 缺少必要的模块。\n详情: {e}")
    sys.exit(1)


parser = argparse.ArgumentParser('Thesis Final V11.1: Exp1 Ablation with Thesis-Aligned Fixes')
parser.add_argument('--data', type=str, default='wikipedia', help='Dataset name')
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
parser.add_argument('--decay_factor', type=float, default=1e-6,
                    help='Time decay factor for structural score. Suggest: 1e-6 for Wiki/Reddit')
parser.add_argument(
    '--ablation',
    type=str,
    default='full',
    choices=['full', 'wostruct', 'baseline', 'wocausal', 'wointerp', 'wogate', 'wotimedecay'],
    help='Ablation setting'
)

try:
    args = parser.parse_args()
except Exception:
    args = parser.parse_args(args=[])


Path('log/').mkdir(parents=True, exist_ok=True)
Path('saved_models/').mkdir(parents=True, exist_ok=True)
Path('results/').mkdir(parents=True, exist_ok=True)


def get_effective_decay(cur_args):
    return 0.0 if cur_args.ablation == 'wotimedecay' else cur_args.decay_factor


def build_engine(cur_args):
    return StreamingGraphEngine(
        window_size_seconds=cur_args.window_size,
        time_decay_factor=get_effective_decay(cur_args)
    )


if args.ablation in ('baseline', 'wostruct') or args.lamda == 0:
    mode_str = f'ABLATION_{args.ablation.upper()}'
else:
    mode_str = f'{args.ablation.upper()}_DECAY{get_effective_decay(args)}_BETA{args.smooth_beta}'

current_time_str = time.strftime('%Y%m%d_%H%M%S')
log_filename = f'log/{args.data}_bs{args.bs}_lr{args.lr}_{mode_str}_{current_time_str}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
logger.info(f'Log file created at: {log_filename}')
logger.info(f'Running Mode: {mode_str}')


def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)


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


class ThesisCTGN(CTGN):
    def __init__(self, num_core_levels, **kwargs):
        self.smooth_beta = kwargs.pop('smooth_beta', 0.2)
        embed_dim = kwargs.get('memory_dimension', 100)
        dropout = kwargs.get('dropout', 0.1)
        self.use_structure = kwargs.pop('use_structure', True)
        self.use_causal = kwargs.pop('use_causal', True)
        self.use_interpolation = kwargs.pop('use_interpolation', True)
        self.use_gate = kwargs.pop('use_gate', True)
        super().__init__(num_communities=1, **kwargs)

        self.structure_encoder = StructureEncoder(embed_dim=embed_dim, max_core=num_core_levels).to(self.device)
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

    def _core_tensor_from_partition(self, nodes, partition):
        if isinstance(partition, dict):
            core_list = [partition.get(str(n), 0) for n in nodes]
            return torch.tensor(core_list, device=self.device, dtype=torch.long)
        if isinstance(partition, torch.Tensor):
            return partition.to(self.device).long()
        return None

    def get_community_embeddings(self, nodes, partition):
        core_tensor = self._core_tensor_from_partition(nodes, partition)
        if core_tensor is None:
            return None
        return self.structure_encoder(core_tensor)

    def _conditional_structure_interpolation(self, self_struct, partner_struct, self_core_tensor):
        """
        论文对齐改动：
        1. 仅在打分阶段执行（compute_temporal_embeddings 本身就是打分阶段）
        2. 仅对冷启动 / 无结构节点执行条件插值
        这里用 core<=0 作为冷启动 / 无有效结构先验的工程判定。
        """
        if (not self.use_interpolation) or self_core_tensor is None:
            return self_struct

        cold_mask = (self_core_tensor <= 0).float().unsqueeze(1)
        beta = self.smooth_beta
        interpolated = (1.0 - beta) * self_struct + beta * partner_struct
        return cold_mask * interpolated + (1.0 - cold_mask) * self_struct

    def _fuse(self, tgn_emb, proj_struct_emb):
        if self.use_gate:
            concat_feat = torch.cat([tgn_emb, proj_struct_emb], dim=1)
            gate = self.gate_net(concat_feat)
            return tgn_emb + gate * proj_struct_emb
        return tgn_emb + proj_struct_emb

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

        src_core_tensor = self._core_tensor_from_partition(source_nodes, partition)
        dst_core_tensor = self._core_tensor_from_partition(destination_nodes, partition)
        neg_core_tensor = self._core_tensor_from_partition(negative_nodes, partition)

        struct_src = self.get_community_embeddings(source_nodes, partition)
        struct_dst = self.get_community_embeddings(destination_nodes, partition)
        struct_neg = self.get_community_embeddings(negative_nodes, partition)

        smooth_src = self._conditional_structure_interpolation(struct_src, struct_dst, src_core_tensor)
        smooth_dst = self._conditional_structure_interpolation(struct_dst, struct_src, dst_core_tensor)
        smooth_neg = self._conditional_structure_interpolation(struct_neg, struct_src, neg_core_tensor)

        proj_struct_src = self.structure_proj(smooth_src)
        proj_struct_dst = self.structure_proj(smooth_dst)
        proj_struct_neg = self.structure_proj(smooth_neg)

        source_embedding = self._fuse(src_base, proj_struct_src)
        destination_embedding = self._fuse(dst_base, proj_struct_dst)
        negative_embedding = self._fuse(neg_base, proj_struct_neg)

        self.last_fused_embeddings = (source_embedding, destination_embedding)
        return source_embedding, destination_embedding, negative_embedding

    def forward_counterfactual(self, neg_hard_cores):
        if self.last_base_embeddings is None:
            raise RuntimeError('Must run normal forward first!')

        _, dst_base = self.last_base_embeddings

        if not self.use_structure:
            return dst_base, torch.ones((dst_base.size(0), 1), device=self.device)

        cf_structure_emb = self.structure_encoder(neg_hard_cores.to(self.device)).detach()
        proj_cf_struct = self.structure_proj(cf_structure_emb).detach()

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


def build_partition(engine, src, dst, neg):
    all_nodes = np.concatenate([src, dst, neg])
    all_cores, _ = engine.get_structure_features(all_nodes)
    return {str(n): int(c) for n, c in zip(all_nodes, all_cores)}


def eval_edge_prediction_thesis(model, negative_edge_sampler, data, n_neighbors, engine, batch_size=200,
                                is_baseline=False):
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

            current_partition = None if is_baseline else build_partition(engine, src, dst, neg)
            pos_prob, neg_prob = model(src, dst, neg, ts, edge_idxs, n_neighbors, partition=current_partition)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_ap), np.mean(val_auc)


def main():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Parameters: {args}')

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        args.data)

    data_feat_dim = node_features.shape[1]
    if args.node_dim != data_feat_dim:
        logger.warning(f'Auto-correcting args.node_dim to {data_feat_dim}...')
        args.node_dim = data_feat_dim
        args.time_dim = data_feat_dim

    engine = build_engine(args)
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    causal_sampler = CausalHardNegativeSampler(full_data.unique_nodes, engine)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False)

    is_baseline_mode = (args.lamda == 0) or (args.ablation == 'baseline')

    if is_baseline_mode:
        logger.info(f'>>> Running BASELINE MODE ({args.ablation}) <<<')
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
            embedding_module_type='graph_attention',
            message_function='identity',
            aggregator_type='last',
            memory_updater_type='gru'
        ).to(device)
    else:
        logger.info(f'>>> Running THESIS MODEL | ablation={args.ablation} <<<')
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
            embedding_module_type='graph_attention',
            message_function='identity',
            aggregator_type='last',
            memory_updater_type='gru',
            use_structure=use_structure,
            use_causal=use_causal,
            use_interpolation=use_interpolation,
            use_gate=use_gate
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    model_save_path = f'saved_models/thesis_final_{args.data}_{args.ablation}.pth'

    num_batch = math.ceil(len(train_data.sources) / args.bs)
    logger.info(f'Start training: {args.n_epoch} epochs')

    for epoch in range(args.n_epoch):
        start_time = time.time()
        model.train()
        model.set_neighbor_finder(train_ngh_finder)
        if args.use_memory:
            model.memory.__init_memory__()

        engine = build_engine(args)
        batch_losses = []

        for i in range(num_batch):
            start_idx = i * args.bs
            end_idx = min((i + 1) * args.bs, len(train_data.sources))
            src = train_data.sources[start_idx:end_idx]
            dst = train_data.destinations[start_idx:end_idx]
            ts = train_data.timestamps[start_idx:end_idx]
            e_idx = train_data.edge_idxs[start_idx:end_idx]

            engine.update_batch(src, dst, ts)
            size = len(src)
            _, neg_rand = train_rand_sampler.sample(size)

            current_partition = None if is_baseline_mode else build_partition(engine, src, dst, neg_rand)

            optimizer.zero_grad()
            pos_prob, neg_prob = model(src, dst, neg_rand, ts, e_idx, args.n_degree, partition=current_partition)
            loss = criterion(pos_prob, torch.ones_like(pos_prob)) + criterion(neg_prob, torch.zeros_like(neg_prob))

            if args.lamda > 0 and (not is_baseline_mode) and getattr(model, 'use_causal', False):
                neg_hard = causal_sampler.sample(size, dst)
                neg_hard_cores, _ = engine.get_structure_features(neg_hard)
                cf_dst_emb, struct_importance = model.forward_counterfactual(
                    torch.tensor(neg_hard_cores, dtype=torch.long)
                )
                reg_loss_val = info_nce_loss(model.last_fused_embeddings[1], cf_dst_emb)
                adaptive_weight = torch.mean(struct_importance)
                loss += args.lamda * adaptive_weight * reg_loss_val

            loss.backward()
            optimizer.step()
            if args.use_memory:
                model.memory.detach_memory()
            batch_losses.append(loss.item())

        model.eval()
        model.set_neighbor_finder(full_ngh_finder)
        val_ap, val_auc = eval_edge_prediction_thesis(
            model, val_rand_sampler, val_data, args.n_degree, engine, args.bs, is_baseline=is_baseline_mode
        )

        logger.info(
            f'Epoch {epoch + 1}: Loss {np.mean(batch_losses):.4f}, Val AP {val_ap:.4f}, '
            f'Time {time.time() - start_time:.2f}s'
        )
        if early_stopper.early_stop_check(val_ap):
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Best model saved to {model_save_path}')
            break

    if args.n_epoch > 0 and not early_stopper.early_stop_check(val_ap):
        torch.save(model.state_dict(), model_save_path)

    logger.info('>>> Starting Final Testing Phase <<<')

    def warmup_engine(data_list):
        new_engine = build_engine(args)
        for d in data_list:
            new_engine.update_batch(d.sources, d.destinations, d.timestamps)
        return new_engine

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    model.eval()
    model.set_neighbor_finder(full_ngh_finder)
    test_engine = warmup_engine([train_data, val_data])
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    test_ap, test_auc = eval_edge_prediction_thesis(
        model, test_rand_sampler, test_data, args.n_degree, test_engine, args.bs, is_baseline=is_baseline_mode
    )

    if args.use_memory:
        model.memory.__init_memory__()
    model.eval()
    model.set_neighbor_finder(full_ngh_finder)
    inductive_engine = warmup_engine([train_data, val_data])
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=42)
    nn_test_ap, nn_test_auc = eval_edge_prediction_thesis(
        model, nn_test_rand_sampler, new_node_test_data, args.n_degree, inductive_engine, args.bs,
        is_baseline=is_baseline_mode
    )

    logger.info('-' * 50)
    logger.info(f'FINAL RESULTS ({args.data} | Mode: {mode_str}):')
    logger.info(f'Transductive Test AP: {test_ap:.4f} | AUC: {test_auc:.4f}')
    logger.info(f'Inductive Test AP:    {nn_test_ap:.4f} | AUC: {nn_test_auc:.4f}')
    logger.info('-' * 50)

    try:
        with open('results/experiment_results.csv', mode='a', newline='') as f:
            csv.writer(f).writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                args.data,
                mode_str,
                args.ablation,
                args.bs,
                args.lr,
                args.decay_factor,
                args.smooth_beta,
                args.lamda,
                f'AP:{test_ap:.4f}',
                f'AUC:{test_auc:.4f}',
                f'In-AP:{nn_test_ap:.4f}',
                f'In-AUC:{nn_test_auc:.4f}'
            ])
    except Exception as e:
        logger.error(f'Save CSV failed: {e}')


if __name__ == '__main__':
    main()
