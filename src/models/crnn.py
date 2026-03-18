import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # (32,128) → (16,64)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # (16,64) → (8,32)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d((2,1)),  # (8,32) → (4,32)

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d((2,1))   # (4,32) → (2,32)
        )

        self.rnn = nn.LSTM(
            input_size=512*2,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 128)

        x = self.cnn(x)
        # (B, 512, 2, 32)

        B, C, H, W = x.size()

        # convert to sequence
        x = x.permute(0, 3, 1, 2)   # (B, W, C, H)
        x = x.contiguous().view(B, W, C*H)  # (B, W, features)

        # RNN
        x, _ = self.rnn(x)

        # FC
        x = self.fc(x)

        # output: (B, W, num_classes)
        return x