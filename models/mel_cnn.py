"""Model 1: small 2D-CNN trained from scratch on log-Mel spectrograms."""
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MelCNN(nn.Module):
    """Input: (B, 1, n_mels, frames) log-Mel. Output: (B, num_classes) logits."""

    def __init__(self, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(1)          # dB values are roughly -80..0
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
