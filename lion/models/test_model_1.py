#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from lion.modules.stacked_rnn import StackedBRNN


class TestModel_1(nn.Module):
    "Basic network using stackedLSTM"
    def __init__(self, args):
        super(TestModel_1, self).__init__()
        print(args['word_dict_size'])
        self.word_emb = nn.Embedding(args['word_dict_size'] + 1, args.word_dim)
        input_dim = args.word_dim
        self.encoder = StackedBRNN(args.word_dim,
                                    args.hidden_size,
                                    args.rnn_layers,
                                    concat_layers=True)
        encoded_dim = args.hidden_size * 2 * args.rnn_layers * 4
        self.classifier = nn.Sequential(
                            nn.Linear(encoded_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, args.classes),
                            nn.LogSoftmax()
                            )

    def forward(self, ex):
        A = self.word_emb(ex['Atoken'])
        B = self.word_emb(ex['Btoken'])
        A = self.encoder(A, ex['Amask'])
        B = self.encoder(B, ex['Bmask'])
        A, _ = torch.max(A, dim=1)
        B, _ = torch.max(B, dim=1)
        merged = torch.cat([A, B, A*B, A-B], dim = 1)
        log_prob = self.classifier(merged)
        return log_prob


