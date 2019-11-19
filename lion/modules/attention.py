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


class XLNetRelativeAttention(nn.Module):
    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions

        if config.d_model % config.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.n_head))

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3]-1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij', attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum('ijbn->bnij', attn_mask)

        # attention probability
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)

        # attention output
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)

        if self.output_attentions:
            return attn_vec, torch.einsum('bnij->ijbn', attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(self, h, g,
                      attn_mask_h, attn_mask_g,
                      r, seg_mat,
                      mems=None, target_mapping=None, head_mask=None):
        if g is not None:
            ###### Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)

            # content-based value head
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # position-based key head
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            ##### h-stream
            # content-stream query head
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            ##### g-stream
            # query-stream query head
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            ###### Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content heads
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)

            # positional heads
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask)

            if self.output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs
