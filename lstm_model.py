import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        padding_idx=0
    ):
        super().__init__()

        # 1. 将字符编号转换为 128 维向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # 2. LSTM 循环神经网络
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. 将最终隐藏状态映射为两个类别的分数
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: [batch_size, sequence_length]
        embedded = self.embedding(input_ids)

        # 忽略补齐部分，避免填充字符影响最终状态
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # hidden 是最终隐藏状态，cell 是最终记忆状态
        _, (hidden, cell) = self.lstm(packed)

        # 取最后一层的隐藏状态
        final_hidden = hidden[-1]

        # 输出形状：[batch_size, 2]
        logits = self.fc(final_hidden)

        return logits
