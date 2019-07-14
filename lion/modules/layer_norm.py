import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Construct a layernorm module (epsilon inside the square root)."""

    def __init__(self, size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.variance_epsilon) + self.bias
