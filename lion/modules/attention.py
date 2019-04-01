#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import div_with_small_value


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, v1, v2, v1_mask=None, v2_mask=None):
        pass


class BasicAttention(Attention):
    """Basic Attention.

    Examples:
         >>> x = BasicAttention()
         >>> v1 = torch.Tensor(1,2,3)
         >>> v2 = torch.Tensor(1,3,3)
         >>> w = torch.Tensor(1,1,1)
         >>> out = x(v1, v2, w)
         >>> assert list(out.size()) == [1, 2, 3]
    """

    def forward(self, v1, v2, v1_mask=None, v2_mask=None):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :return: (batch, seq_len1, seq_len2)
        """

        # (batch, seq_len1, 1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        # (batch, 1, seq_len2)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        # (batch, seq_len1, seq_len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        d = v1_norm * v2_norm

        return div_with_small_value(a, d)


class SoftmaxAttention(Attention):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self, v1, v2, v1_mask=None, v2_mask=None):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        prem_hyp_attn = F.softmax(similarity_matrix.masked_fill(v1_mask.unsqueeze(2), -float('inf')), dim=1)
        hyp_prem_attn = F.softmax(similarity_matrix.masked_fill(v2_mask.unsqueeze(1), -float('inf')), dim=2)

        attended_premises = hyp_prem_attn.bmm(v2)
        attended_hypotheses = prem_hyp_attn.transpose(1, 2).bmm(v1)

        attended_premises.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_hypotheses.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_premises, attended_hypotheses
