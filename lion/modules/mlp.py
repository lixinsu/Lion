import torch
import torch.nn as nn

from lion.modules.utils import gelu


class MLP(nn.Module):
    """Used for Transformer."""
    def __init__(self, hidden_size, output_size, act='relu'):
        super(MLP, self).__init__()
        self.dense = nn.Linear(hidden_size, output_size)
        if act == 'relu':
            self.intermediate_act_fn = torch.nn.functional.relu
        else:
            self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        # b*s*hidden_size
        hidden_states = self.dense(hidden_states)
        # b*s*intermediate_size
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
