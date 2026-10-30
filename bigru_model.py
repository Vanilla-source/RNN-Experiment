import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiGRUModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        padding_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 双向 GRU 的最终表示维度为 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.gru(packed)

        # hidden: [2, batch_size, hidden_dim]
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # 拼接正向和反向的最终隐藏状态
        representation = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        logits = self.fc(representation)
        return logits
