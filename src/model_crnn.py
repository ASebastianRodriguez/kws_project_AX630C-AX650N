from __future__ import annotations
import torch
import torch.nn as nn

NUM_CLASSES = 8

class CRNNSmall(nn.Module):
    def __init__(self, n_mels: int = 64, num_classes: int = NUM_CLASSES, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.gru = nn.GRU(input_size=32*(n_mels//4), hidden_size=hidden,
                          batch_first=True, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(2*hidden, num_classes)

    def forward(self, x):           # x: [B,1,T,F]
        x = self.conv(x)            # [B,32,T/4,F/4]
        B, C, T2, F2 = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B, T2, C*F2)  # [B,T2, 32*F/4]
        y, _ = self.gru(x)          # [B,T2, 2*hidden]
        y = y[:, -1, :]             # último paso temporal
        return self.fc(y)           # [B, num_classes]
