
import torch
from rnn_model import RNNModel

# 模拟词表大小为 1000 的情况
model = RNNModel(vocab_size=1000)

# 模拟一个批次：64 条评论，每条长度为 200
input_ids = torch.randint(2, 1000, (64, 200))

# 模拟每条评论的实际长度，必须大于 0
lengths = torch.randint(1, 201, (64,))

# 前向传播
logits = model(input_ids, lengths)

print("模型测试通过！")
print("输出形状：", logits.shape)
print("参数总量：", sum(p.numel() for p in model.parameters()))
