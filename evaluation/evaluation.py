# 文件路径: evaluation/evaluation.py (最终“在线评估”版)

import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200, communities=None,
                         partition_end_times=None):
    """
    Evaluate the edge prediction task using the "online" protocol.
    """
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
            sources_batch = data.sources[start_idx:end_idx]
            destinations_batch = data.destinations[start_idx:end_idx]
            timestamps_batch = data.timestamps[start_idx:end_idx]
            edge_idxs_batch = data.edge_idxs[start_idx:end_idx]

            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)

            # 寻找当前批次对应的社群快照 (逻辑与训练脚本一致)
            current_partition = None
            if communities is not None:
                current_ts = np.mean(timestamps_batch)
                partition_idx = np.searchsorted(partition_end_times, current_ts)
                partition_idx = min(partition_idx, len(communities) - 1)
                current_partition = communities[partition_idx]

            # --- 【最终核心修正：对齐官方评估协议】---
            # 调用我们为“在线”评估专门创建的新接口
            # 这个接口内部会处理“先更新内存，后预测负样本”的逻辑
            pos_prob, neg_prob = model.compute_edge_probabilities_online(sources_batch,
                                                                         destinations_batch,
                                                                         negatives_batch,
                                                                         timestamps_batch,
                                                                         edge_idxs_batch,
                                                                         n_neighbors,
                                                                         partition=current_partition)
            # --- 【修正结束】---

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    """
    Evaluate the node classification task (保持不变).
    """
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn = tgn.eval()
        decoder = decoder.eval()
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_instance, start_idx + batch_size)

            sources_batch = data.sources[start_idx:end_idx]
            destinations_batch = data.destinations[start_idx:end_idx]
            timestamps_batch = data.timestamps[start_idx:end_idx]
            edge_idxs_batch = edge_idxs[start_idx:end_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)

            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[start_idx:end_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc