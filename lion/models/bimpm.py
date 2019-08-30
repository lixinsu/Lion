#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from lion.modules.layer_match import FullLayerMatch, MaxPoolingLayerMatch
from lion.modules.attention import BasicAttention
from lion.modules.utils import div_with_small_value


class BIMPM(nn.Module):
    """
    Implementation of the BIMPM model presented in the paper "Bilateral Multi-Perspective
    Matching for Natural Language Sentences" by Wang et al.
    """
    MODEL_DEFAULTS = {'max_word_length': 10,
                        'dropout': 0.1,
                        'num_perspective': 20,
                        'use_char_emb': True,
                        'char_hidden_size': 50,
                        'char_dim': 20,
                        'hidden_size': 100,
                        'rnn_layers': 2,
                        'word_dim': 300}

    def __init__(self, args):
        super(BIMPM, self).__init__()
        self.args = args
        self.fill_default_parameters()
        self.input_size = self.args['word_dim'] + int(self.args['use_char_emb']) * self.args['char_hidden_size']
        self.num_perspective = self.args['num_perspective']

        # Word Representation Layer
        self.char_emb = nn.Embedding(args['char_dict_size'],args['char_dim'], padding_idx=0)
        self.word_embedding = nn.Embedding(args['word_dict_size'], args['word_dim'])
        # no fine-tuning for word vectors
        self.word_embedding.weight.requires_grad = False
        self.full_match = FullLayerMatch()
        self.max_pooling_match = MaxPoolingLayerMatch()
        self.attention= BasicAttention()

        self.char_LSTM = nn.LSTM(
            input_size=self.args['char_dim'],
            hidden_size=self.args['char_hidden_size'],
            num_layers=1,
            bidirectional=False,
            batch_first=True)

        # Context Representation Layer
        self.context_LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.args['hidden_size'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Matching Layer
        for i in range(1, 9):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.num_perspective, self.args['hidden_size'])))

        # Aggregation Layer
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.num_perspective * 8,
            hidden_size=self.args['hidden_size'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        # Prediction Layer
        self.pred_fc1 = nn.Linear(self.args['hidden_size'] * 4, self.args['hidden_size'] * 2)
        self.pred_fc2 = nn.Linear(self.args['hidden_size'] * 2, self.args['classes'])

        self.reset_parameters()

    def fill_default_parameters(self):
        for k, v in BIMPM.MODEL_DEFAULTS.items():
            if k not in self.args:
                self.args.update({k: v})

    def reset_parameters(self):
        # Word Representation Layer
        nn.init.uniform_(self.char_emb.weight, -0.005, 0.005)
        # zero vectors for padding
        self.char_emb.weight.data[0].fill_(0)

        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.word_embedding.weight.data[0], -0.1, 0.1)

        nn.init.kaiming_normal_(self.char_LSTM.weight_ih_l0)
        nn.init.constant_(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.char_LSTM.weight_hh_l0)
        nn.init.constant_(self.char_LSTM.bias_hh_l0, val=0)

        # Context Representation Layer
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # Matching Layer
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)

        # Aggregation Layer
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # Prediction Layer
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        self.d = nn.Dropout(p=self.args['dropout'])
        return self.d(v)

    def forward(self, ex):
        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        A = self.word_embedding(ex['Atoken'])
        B = self.word_embedding(ex['Btoken'])
        Amask = ex['Amask'].unsqueeze(2).float()
        Bmask = ex['Bmask'].unsqueeze(2).float()


        if self.args.use_char_emb:
            # (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
            seq_len_A = ex['Achar'].size(1)
            seq_len_B = ex['Bchar'].size(1)

            char_A = ex['Achar'].view(-1, self.args['max_word_length'])
            char_B = ex['Bchar'].view(-1, self.args['max_word_length'])

            # (batch * seq_len, max_word_len, char_dim)-> (1, batch * seq_len, char_hidden_size)
            _, (char_A, _) = self.char_LSTM(self.char_emb(char_A))
            _, (char_B, _) = self.char_LSTM(self.char_emb(char_B))

            # (batch, seq_len, char_hidden_size)
            char_A = char_A.view(-1, seq_len_A, self.args['char_hidden_size'])
            char_B = char_B.view(-1, seq_len_B, self.args['char_hidden_size'])

            # (batch, seq_len, word_dim + char_hidden_size)
            A = torch.cat([A, char_A], dim=-1)
            B = torch.cat([B, char_B], dim=-1)

        A = self.dropout(A)
        B = self.dropout(B)

        # Context Representation Layer
        # (batch, seq_len, hidden_size * 2)
        context_A, _ = self.context_LSTM(A)
        context_B, _ = self.context_LSTM(B)

        context_A = self.dropout(context_A*Amask)
        context_B = self.dropout(context_B*Bmask)

        # (batch, seq_len, hidden_size)
        context_A_fw, context_A_bw = torch.split(context_A, self.args['hidden_size'], dim=-1)
        context_B_fw, context_B_bw = torch.split(context_B, self.args['hidden_size'], dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_A_full_fw = self.full_match(context_A_fw, context_B_fw[:, -1, :], self.mp_w1)
        mv_A_full_bw = self.full_match(context_A_bw, context_B_bw[:, 0, :], self.mp_w2)
        mv_B_full_fw = self.full_match(context_B_fw, context_A_fw[:, -1, :], self.mp_w1)
        mv_B_full_bw = self.full_match(context_B_bw, context_A_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = self.max_pooling_match(context_A_fw, context_B_fw, self.mp_w3)
        mv_max_bw = self.max_pooling_match(context_A_bw, context_B_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_A_max_fw, _ = mv_max_fw.max(dim=2)
        mv_A_max_bw, _ = mv_max_bw.max(dim=2)
        mv_B_max_fw, _ = mv_max_fw.max(dim=1)
        mv_B_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = self.attention(context_A_fw, context_B_fw)
        att_bw = self.attention(context_A_bw, context_B_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_B_fw = context_B_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_B_bw = context_B_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_A_fw = context_A_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_A_bw = context_A_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_B_fw = div_with_small_value(att_B_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_B_bw = div_with_small_value(att_B_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_A_fw = div_with_small_value(att_A_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_A_bw = div_with_small_value(att_A_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, l)
        mv_A_att_mean_fw = self.full_match(context_A_fw, att_mean_B_fw, self.mp_w5)
        mv_A_att_mean_bw = self.full_match(context_A_bw, att_mean_B_bw, self.mp_w6)
        mv_B_att_mean_fw = self.full_match(context_B_fw, att_mean_A_fw, self.mp_w5)
        mv_B_att_mean_bw = self.full_match(context_B_bw, att_mean_A_bw, self.mp_w6)

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_B_fw, _ = att_B_fw.max(dim=2)
        att_max_B_bw, _ = att_B_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_A_fw, _ = att_A_fw.max(dim=1)
        att_max_A_bw, _ = att_A_bw.max(dim=1)

        # (batch, seq_len, l)
        mv_A_att_max_fw = self.full_match(context_A_fw, att_max_B_fw, self.mp_w7)
        mv_A_att_max_bw = self.full_match(context_A_bw, att_max_B_bw, self.mp_w8)
        mv_B_att_max_fw = self.full_match(context_B_fw, att_max_A_fw, self.mp_w7)
        mv_B_att_max_bw = self.full_match(context_B_bw, att_max_A_bw, self.mp_w8)

        # (batch, seq_len, l * 8)
        mv_A = torch.cat(
            [mv_A_full_fw, mv_A_max_fw, mv_A_att_mean_fw, mv_A_att_max_fw,
             mv_A_full_bw, mv_A_max_bw, mv_A_att_mean_bw, mv_A_att_max_bw], dim=2)
        mv_B = torch.cat(
            [mv_B_full_fw, mv_B_max_fw, mv_B_att_mean_fw, mv_B_att_max_fw,
             mv_B_full_bw, mv_B_max_bw, mv_B_att_mean_bw, mv_B_att_max_bw], dim=2)

        mv_A = self.dropout(mv_A)*Amask
        mv_B = self.dropout(mv_B)*Bmask

        # Aggregation Layer
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_A_last, _) = self.aggregation_LSTM(mv_A)
        _, (agg_B_last, _) = self.aggregation_LSTM(mv_B)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        output = torch.cat(
            [agg_A_last.permute(1, 0, 2).contiguous().view(-1, self.args['hidden_size'] * 2),
             agg_B_last.permute(1, 0, 2).contiguous().view(-1, self.args['hidden_size'] * 2)], dim=1)
        output = self.dropout(output)

        # Prediction Layer
        output = torch.tanh(self.pred_fc1(output))
        output = self.dropout(output)
        logits = self.pred_fc2(output)

        return logits
