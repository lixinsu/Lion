#!/usr/bin/env python
# coding: utf-8
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import div_with_small_value


class Attention(nn.Module):
    """Attention weights are computed by q, k and v, since k and v are always the same,
    so here we implement this for simple."""
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, v1, v2, v1_mask=None, v2_mask=None):
        raise NotImplementedError


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
        # batch matrix-matrix product of matrices
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
            v1: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            v1_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            v2: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            v2_mask: A mask for the sequences in the hypotheses batch,
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


class MultiHeadedAttention(Attention):
    """Multi-head attention allows the model to jointly attend to information from different
     representation subspaces at different positions."""
    def __init__(self, num_head, dim_model, dropout_prob=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.num_attention_heads = num_head
        self.attention_head_size = int(dim_model / num_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(dim_model, self.all_head_size)
        self.key = nn.Linear(dim_model, self.all_head_size)
        self.value = nn.Linear(dim_model, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def split_heads(self, x):
        # batch_size*sequence*num_attention_heads*attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # batch_size*num_attention_heads*sequence*attention_head_size
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        # new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        new_x_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def forward(self, v1, v2=None, v1_mask=None, v2_mask=None):
        mixed_query_layer = self.query(v1)
        mixed_key_layer = self.key(v1)
        mixed_value_layer = self.value(v1)

        # batch_size*num_attention_heads*sequence*attention_head_size
        query_layer = self.split_heads(mixed_query_layer)
        key_layer = self.split_heads(mixed_key_layer)
        value_layer = self.split_heads(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # batch_size*num_attention_heads*sequence*sequence
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask: [batch_size, 1, 1, sequence_length]
        attention_scores = attention_scores + v1_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # attention_probs: batch_size*num_attention_heads*sequence*sequence
        # value_layer: batch_size*num_attention_heads*sequence*attention_head_size
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.merge_heads(context_layer)

        return context_layer
