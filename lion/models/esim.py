#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

from lion.modules.dropout import RNNDropout
#from lion.modules.seq2seq_encoder import Seq2SeqEncoder
from lion.modules.stacked_rnn import StackedBRNN
from lion.modules.attention import SoftmaxAttention


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self, args):
        """
        Args:
            word_dict_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
        """
        super(ESIM, self).__init__()

        self.vocab_size = args['word_dict_size']
        self.embedding_dim = args['word_dim']
        self.hidden_size = args['hidden_size']
        self.num_classes = args['classes']
        self.dropout = args['dropout']

        self._word_embedding = nn.Embedding(self.vocab_size + 1,
                                            self.embedding_dim,
                                            padding_idx=0,
                                            _weight=None)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = StackedBRNN(self.embedding_dim, self.hidden_size, 1,
                            dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                            concat_layers=False, padding=False)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = StackedBRNN(self.hidden_size, self.hidden_size, 1,
                            dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                            concat_layers=False, padding=False)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self, ex):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises = ex['Atoken']
        hypotheses = ex['Btoken']
        premises_mask = ex['Amask']
        hypotheses_mask = ex['Bmask']

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_mask)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_mask)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_mask)
        v_bj = self._composition(projected_hypotheses, hypotheses_mask)

        reversed_premises_mask = (1-premises_mask).float()
        reversed_hypotheses_mask = (1 - hypotheses_mask).float()

        v_a_avg = torch.sum(v_ai * reversed_premises_mask.unsqueeze(2), dim=1)\
            / torch.sum(reversed_premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * reversed_hypotheses_mask.unsqueeze(2), dim=1)\
            / torch.sum(reversed_hypotheses_mask, dim=1, keepdim=True)

        v_ai = v_ai.masked_fill(premises_mask.unsqueeze(2), -1e7)
        v_bj = v_bj.masked_fill(hypotheses_mask.unsqueeze(2), -1e7)

        v_a_max, _ = v_ai.max(dim=1)
        v_b_max, _ = v_bj.max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        return log_prob


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

