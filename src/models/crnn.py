import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # (32,128) -> (16,64)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # (16,64) -> (8,32)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # (8,32) -> (4,32)

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # (4,32) -> (2,32)
        )

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=512 * 2,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
