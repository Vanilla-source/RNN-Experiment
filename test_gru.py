
import torch
from gru_model import GRUModel

model = GRUModel(vocab_size=1000)

# 模拟 64 条评论，每条最多 200 个字符
input_ids = torch.randint(2, 1000, (64, 200))
lengths = torch.randint(1, 201, (64,))

logits = model(input_ids, lengths)

print("GRU 模型测试通过！")
print("输出形状：", logits.shape)
print("参数总量：", sum(p.numel() for p in model.parameters()))
