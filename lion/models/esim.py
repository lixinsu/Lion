#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from lion.modules.dropout import RNNDropout
from lion.modules.stacked_rnn import StackedBRNN
from lion.modules.attention import DotProductAttention, masked_softmax


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    MODEL_DEFAULTS = {'dropout': 0.1,
                      'hidden_size': 100,
                      'word_dim': 300}

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
        self.args = args
        self.fill_default_parameters()
        self.vocab_size = args['word_dict_size']
        self.embedding_dim = args['word_dim']
        self.hidden_size = args['hidden_size']
        self.num_classes = args['classes']
        self.dropout = args['dropout']
        self.word_embedding = nn.Embedding(self.vocab_size + 1,
                                           self.embedding_dim,
                                           padding_idx=0,
                                           _weight=None)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = StackedBRNN(self.embedding_dim, self.hidden_size, 1,
                                     dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                                     concat_layers=False, padding=False)

        self._attention = DotProductAttention()

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

    def fill_default_parameters(self):
        for k, v in ESIM.MODEL_DEFAULTS.items():
            if k not in self.args:
                self.args.update({k: v})

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
        premises = ex['Atoken_ids']
        hypotheses = ex['Btoken_ids']
        premises_mask = ex['Amask']
        hypotheses_mask = ex['Bmask']
        Amask = torch.ByteTensor(premises.size(0), premises.size(1)).fill_(1)
        for i, d in enumerate(premises_mask):
            Amask[i, :d.sum()].fill_(0)
        Bmask = torch.ByteTensor(hypotheses_mask.size(0), hypotheses_mask.size(1)).fill_(1)
        for i, d in enumerate(hypotheses_mask):
            Bmask[i, :d.sum()].fill_(0)
        premises_mask = Amask.cuda()
        hypotheses_mask = Bmask.cuda()

        if self.args['use_elmo']:
            elmo_premises = ex['Atoken']
            elmo_hypotheses = ex['Btoken']
            if self.args['use_elmo'] == 'only':
                embedded_premises = self.word_embedding(elmo_premises)['elmo_representations'][0]
                embedded_hypotheses = self.word_embedding(elmo_hypotheses)['elmo_representations'][0]
            elif self.args['use_elmo'] == 'concat':
                embedded_premises = self.word_embedding(premises)
                embedded_hypotheses = self.word_embedding(hypotheses)
                elmo_embedded_premises = self.elmo_embedding(elmo_premises)['elmo_representations'][0]
                elmo_embedded_hypotheses = self.elmo_embedding(elmo_hypotheses)['elmo_representations'][0]
                embedded_premises = torch.cat((embedded_premises, elmo_embedded_premises), dim=-1)
                embedded_hypotheses = torch.cat((embedded_hypotheses, elmo_embedded_hypotheses), dim=-1)
            else:
                raise ValueError('Invalid argument of [use_elmo] :{}'.format(self.args['use_elmo']))
        else:
            embedded_premises = self.word_embedding(premises)
            embedded_hypotheses = self.word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_mask.bool())
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_mask.bool())

        attention_weight = self._attention(encoded_premises, encoded_hypotheses,
                            premises_mask.bool(), hypotheses_mask.bool())
        premises_att_weigth = masked_softmax(attention_weight, premises_mask.bool(),  dim=1)
        hypotheses_att_weigth = masked_softmax(attention_weight, hypotheses_mask.bool())

        attended_premises = hypotheses_att_weigth.bmm(encoded_hypotheses)
        attended_hypotheses = premises_att_weigth.transpose(1, 2).bmm(encoded_premises)

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

        v_ai = self._composition(projected_premises, premises_mask.bool())
        v_bj = self._composition(projected_hypotheses, hypotheses_mask.bool())

        reversed_premises_mask = (1-premises_mask).float()
        reversed_hypotheses_mask = (1 - hypotheses_mask).float()

        v_a_avg = torch.sum(v_ai * reversed_premises_mask.unsqueeze(2), dim=1)\
            / torch.sum(reversed_premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * reversed_hypotheses_mask.unsqueeze(2), dim=1)\
            / torch.sum(reversed_hypotheses_mask, dim=1, keepdim=True)

        v_ai = v_ai.masked_fill(premises_mask.unsqueeze(2).bool(), -1e7)
        v_bj = v_bj.masked_fill(hypotheses_mask.unsqueeze(2).bool(), -1e7)

        v_a_max, _ = v_ai.max(dim=1)
        v_b_max, _ = v_bj.max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        return logits


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

