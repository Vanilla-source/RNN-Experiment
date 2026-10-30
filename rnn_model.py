import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        padding_idx=0
    ):
        super(RNNModel, self).__init__()

        # 1. 将字符 ID 转换为词向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # 2. 构建普通 RNN
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. 将最终隐藏状态映射为两个类别的分数
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: [batch_size, max_len]
        # lengths: [batch_size]

        # [batch_size, max_len, embed_dim]
        embedded = self.embedding(input_ids)

        # 忽略补齐位置，避免模型把 padding 当作有效文本
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # hidden: [1, batch_size, hidden_dim]
        _, hidden = self.rnn(packed)

        # 取最后一层的最终隐藏状态
        # [batch_size, hidden_dim]
        final_hidden = hidden[-1]

        # [batch_size, 2]
        logits = self.fc(final_hidden)

        return logits
