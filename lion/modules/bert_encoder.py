import copy

import torch.nn as nn

from lion.modules.mlp import MLP
from lion.modules.layer_norm import LayerNorm
from lion.modules.attention import MultiHeadedAttention


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = SubLayer(config.num_attention_heads, config.hidden_size, config.intermediate_size,
                         config.hidden_dropout_prob, config.hidden_act, config.layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # embedding_output, extended_attention_mask,output_all_encoded_layers
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SubLayer(nn.Module):
    def __init__(self, num_head, dim_model, dim_out, dropout_prob=0.1, act='relu', eps=1e-12):
        super(SubLayer, self).__init__()
        self.attention = BertAttention(num_head, dim_model, dropout_prob, eps)
        self.intermediate = MLP(dim_model, dim_out, act)
        self.output = EncoderOutput(dim_out, dim_model, dropout_prob, eps)

    def forward(self, hidden_states, attention_mask):
        # embedding_output, extended_attention_mask
        # attention_output.size == embedding_output.size == b*s*h
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, num_head, dim_model, dropout_prob, eps):
        super(BertAttention, self).__init__()
        self.self = MultiHeadedAttention(num_head, dim_model, dropout_prob)
        self.output = SublayerOutput(dim_model, dropout_prob, eps)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(v1=input_tensor, v1_mask=attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class SublayerOutput(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, hidden_size, dropout, eps):
        super(SublayerOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        """"Apply residual connection to any sublayer with the same size."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderOutput(nn.Module):
    def __init__(self, in_size, out_size, dropout, eps):
        super(EncoderOutput, self).__init__()
        self.dense = nn.Linear(in_size, out_size)
        self.LayerNorm = LayerNorm(out_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        # b*s*hidden_size
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
