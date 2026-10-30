
import torch
from lstm_model import LSTMModel

model = LSTMModel(vocab_size=1000)

# 模拟一个包含 64 条评论的批次
input_ids = torch.randint(2, 1000, (64, 200))
lengths = torch.randint(1, 201, (64,))

logits = model(input_ids, lengths)

print("LSTM 模型测试通过！")
print("输出形状：", logits.shape)
print("参数总量：", sum(p.numel() for p in model.parameters()))
