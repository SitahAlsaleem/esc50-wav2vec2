"""Models 2 and 3: trainable heads on top of frozen Wav2Vec 2.0 frame features.

Input: (B, T, 768) frame features.  Output: (B, num_classes) logits.
"""
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Model 2: mean-pool over time, then a linear classifier."""

    def __init__(self, dim: int = 768, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x.mean(dim=1))
        return self.fc(self.drop(x))


class AttentionHead(nn.Module):
    """Model 3: temporal attention pooling.

    score_t = v^T tanh(W h_t + b);  a = softmax_t(score);  z = sum_t a_t h_t
    The learned weights a_t say which frames matter (plotted in the demo).
    """

    def __init__(self, dim: int = 768, hidden: int = 256, num_classes: int = 50,
                 dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_attention: bool = False):
        x = self.norm(x)
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)       # (B, T)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=1)                 # (B, dim)
        logits = self.fc(self.drop(pooled))
        return (logits, attn) if return_attention else logits


class UniformHead(nn.Module):
    """Ablation control for Model 3.

    Identical to AttentionHead (per-frame LayerNorm -> weighted sum over time ->
    dropout -> linear) except the scoring network is removed and every frame gets
    the same weight 1/T. Comparing it with AttentionHead isolates the learned
    attention weights; comparing it with LinearHead isolates the order of
    normalisation and pooling.
    """

    def __init__(self, dim: int = 768, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x, return_attention: bool = False):
        x = self.norm(x)
        pooled = x.mean(dim=1)
        logits = self.fc(self.drop(pooled))
        if return_attention:
            attn = torch.full(x.shape[:2], 1.0 / x.size(1), device=x.device)
            return logits, attn
        return logits
