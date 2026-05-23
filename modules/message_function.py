"""
把一条交互事件，变成一条 message
"""
from torch import nn


"""
抽象基类
"""
class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None

"""
输入维度：raw_message_dimension
输出维度：message_dimension
网络结构：先线性降到一半维度,ReLU,再映射到目标 message 维度
"""
class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages

"""
消息函数什么都不做，直接把 raw message 当最终 message。
不额外学习“如何重组这些信息”
直接把拼接结果交给 memory updater（如 GRU）
"""
class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):

    return raw_messages

"""
message function 可选择使用"mlp"或者"identity"，方便做 ablation study（消融实验）
identity  “让 raw message 本身承担表达，后续 GRU 自己学更新。”
mlp  先把交互事件编码成更适合写入 memory 的格式，再更新 memory。
"""
def get_message_function(module_type, raw_message_dimension, message_dimension):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "identity":
    return IdentityMessageFunction()
