"""
如果一个节点在同一个 batch 更新前收到了多条 message，该保留哪条，
或者怎么合并,用 aggregator 变成一条最终更新消息,再送给 GRU 更新 memory
"""
from collections import defaultdict
import torch
import numpy as np

"""
抽象基类
输入：一批 node_ids
输入：对应的 messages
输出：把相同 node id 的多条消息聚合起来
"""
class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """
  """
  做分桶：把输入从“按事件排列”变成“按节点编号归类”
  """
  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


"""
对每个节点,如果它有多条 message,只保留最后一条（最新一条）message,timestamp 也保留最后一条的时间戳
"""
class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

"""
同一个节点的所有 message 都拿出来,先stack,然后按维度求平均,时间戳仍然取最后一条消息的时间
"""
class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


"""
可选择使用"last"或者"mean"，方便做 ablation study（消融实验）
"""
def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
