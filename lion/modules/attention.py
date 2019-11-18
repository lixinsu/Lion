#!/usr/bin/env python
# coding: utf-8
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Attention weights are computed by q, k and v, since k and v are always the same,
    so here we implement this for simple."""
    def __init__(self, normalize=False):
        super().__init__()
        self._normalize = normalize

    def forward(self, q, k, q_mask=None, k_mask=None):
        raise NotImplementedError


class DotProductAttention(Attention):
    """Dot Product style for computing similarity between q and k."""
    def forward(self, q, k, q_mask=None, k_mask=None):
        similarities = torch.bmm(q, k.permute(0, 2, 1))

        if self._normalize:
            # normalize q as default
            return masked_softmax(similarities, q_mask, dim=1)

        return similarities


class CosineAttention(Attention):
    """Cosine style for computing similarity between q and k."""

    def forward(self, q, k, q_mask=None, k_mask=None):
        q_norm = q.norm(p=2, dim=-1, keepdim=True) + 1e-13
        k_norm = (k.norm(p=2, dim=-1, keepdim=True) + 1e-13).permute(0, 2, 1)
        similarities = torch.bmm(q, k.permute(0, 2, 1)) / (q_norm * k_norm)

        if self._normalize:
            # normalize q as default
            return masked_softmax(similarities, q_mask, dim=1)

        return similarities


class BilinearAttention(Attention):
    """Bilinear style for computing similarity between q and k."""
    def __init__(self, embedding_dim, activation=None, normalize=False):
        super().__init__(normalize)
        self._weight_matrix = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self._bias = torch.nn.Parameter(torch.Tensor(1))
        self._activation = activation if not activation else lambda: lambda x: x,
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, q, k, q_mask=None, k_mask=None):
        intermediate = q.mm(self._weight_matrix)
        similarities = self._activation(intermediate.bmm(k.transpose(1, 2)) + self._bias)

        if self._normalize:
            # normalize q as default
            return masked_softmax(similarities, q_mask, dim=1)

        return similarities


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

    def forward(self, q, k=None, q_mask=None, k_mask=None):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(q)
        mixed_value_layer = self.value(q)

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
        attention_scores = attention_scores + q_mask

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


def masked_softmax(similarity, mask, dim=-1):
    if mask is None:
        masked_softmax_similarity = F.softmax(similarity, dim=dim)
    else:
        mask = mask.float()
        if mask.dim() < similarity.dim() and dim == 1:
            # compute q's attention weight
            mask = mask.unsqueeze(2)
        elif mask.dim() < similarity.dim() and dim != 1:
            # compute k's attention weight
            mask = mask.unsqueeze(1)

        masked_vector = similarity.masked_fill(mask.to(dtype=torch.bool), -1e32)
        masked_softmax_similarity = torch.nn.functional.softmax(masked_vector, dim=dim)

    return masked_softmax_similarity
