#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from lion.nn.utils import div_with_small_value


class LayerMatch(nn.Module):
    def __init__(self):
        super(LayerMatch, self).__init__()

    def forward(self, v1, v2, w):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l)
        """
        pass


class FullLayerMatch(LayerMatch):
    """Match layer.

    Examples:
        >>> layer=FullLayerMatch()
        >>> v1 = torch.Tensor(1,2,3)
        >>> v2 = torch.Tensor(1,3)
        >>> w = torch.Tensor(4,3)
        >>> out = layer(v1, v2, w)
        >>> assert list(out.size()) == [1, 2, 4]

    """

    def forward(self, v1, v2, w):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, seq_len, l)
        """
        seq_len = v1.size(1)

        # Trick for large memory requirement
        """
        if len(v2.size()) == 2:
            v2 = torch.stack([v2] * seq_len, dim=1)

        m = []
        for i in range(self.l):
            # v1: (batch, seq_len, hidden_size)
            # v2: (batch, seq_len, hidden_size)
            # w: (1, 1, hidden_size)
            # -> (batch, seq_len)
            m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))

        # list of (batch, seq_len) -> (batch, seq_len, l)
        m = torch.stack(m, dim=2)
        """

        # (1, 1, hidden_size, l)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        length = w.size(0)
        v1 = w * torch.stack([v1] * length, dim=3)
        if len(v2.size()) == 3:
            v2 = w * torch.stack([v2] * length, dim=3)
        else:
            v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * length, dim=3)

        m = F.cosine_similarity(v1, v2, dim=2)

        return m


class MaxPoolingLayerMatch(LayerMatch):
    """Max Pooling Match layer.

    Examples:
        >>> layer=MaxPoolingLayerMatch()
        >>> v1 = torch.Tensor(1,2,3)
        >>> v2 = torch.Tensor(1,5,3)
        >>> w = torch.Tensor(4,3)
        >>> out = layer(v1, v2, w)
        >>> assert list(out.size()) == [1, 2, 5, 4]

    """
    def forward(self, v1, v2, w):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, seq_len1, seq_len2, l)
        """

        # Trick for large memory requirement
        """
        m = []
        for i in range(self.l):
            # (1, 1, hidden_size)
            w_i = w[i].view(1, 1, -1)
            # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
            v1, v2 = w_i * v1, w_i * v2
            # (batch, seq_len, hidden_size->1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True)

            # (batch, seq_len1, seq_len2)
            n = torch.matmul(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm.permute(0, 2, 1)

            m.append(div_with_small_value(n, d))

        # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
        m = torch.stack(m, dim=3)
        """

        # (1, l, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)
        length = w.size(0)
        # (batch, l, seq_len, hidden_size)
        v1, v2 = w * torch.stack([v1] * length, dim=1), w * torch.stack([v2] * length, dim=1)
        # (batch, l, seq_len, hidden_size->1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)

        # (batch, l, seq_len1, seq_len2)
        n = torch.matmul(v1, v2.transpose(2, 3))
        d = v1_norm * v2_norm.transpose(2, 3)

        # (batch, seq_len1, seq_len2, l)
        m = div_with_small_value(n, d).permute(0, 2, 3, 1)

        return m