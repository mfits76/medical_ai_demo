"""TextCNN specialty classifier in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialtyTextCNN(nn.Module):
    """Kim (2014)-style TextCNN for multi-class clinical note triage.

    Compact and fast to train — suitable as an interview PoC that still
    demonstrates real PyTorch modeling choices (embeddings, conv banks,
    dropout, softmax over specialties).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.4,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq)
        x = self.embedding(input_ids)  # (batch, seq, embed)
        x = x.transpose(1, 2)  # (batch, embed, seq)
        features = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.dropout(x)
        return self.fc(x)
