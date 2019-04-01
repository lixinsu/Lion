#!/usr/bin/env python
# coding: utf-8

import yaml


rv = {}

# Path
rv['meta_dir'] = 'data/preprocessed/QQP/'
rv['train_file'] = 'data/preprocessed/QQP/train_spacy.jsonl'
rv['dev_file'] = 'data/preprocessed/QQP/train_spacy.jsonl'
rv['test_file'] = 'data/preprocessed/QQP/test_spacy.jsonl'
rv['result_dir'] = 'data/outputs/QQP/'
rv['embedding_file'] = '/home/fanyixing/users/mxy/coqa-baselines/wordvecs/glove.42B.300d.txt'
# rv['meta_dir'] = 'data/preprocessed/QQPdebug/'
# rv['train_file'] = 'data/preprocessed/QQPdebug/train_spacy.jsonl'
# rv['dev_file'] = 'data/preprocessed/QQPdebug/train_spacy.jsonl'
# rv['test_file'] = 'data/preprocessed/QQPdebug/test_spacy.jsonl'
# rv['result_dir'] = 'data/outputs/QQPdebug/'
# rv['embedding_file'] = ''

# General
rv['batch_size'] = 25
rv['epoches'] = 20
rv['num_workers'] = 2
rv['use_cuda'] = True

# Model
rv['max_A_len'] = 256
rv['max_B_len'] = 256
rv['max_word_length'] = 16
rv['dropout'] = 0.1
rv['num_perspective'] = 20
rv['use_char_emb'] = True
rv['char_hidden_size'] = 50
rv['network'] = 'bimpm'
rv['char_dim'] = 20
rv['hidden_size'] = 100
rv['rnn_layers'] = 2
rv['classes'] = 3
rv['word_dim'] = 300
rv['min_cnt'] = 0
rv['fix_embeddings'] = False
rv['optimizer'] = 'adamax'
rv['weight_decay'] = 0
rv['grad_clipping'] = 10

# Vocab信息后更新


yaml.dump(rv, open('lion/configs/test_bimpm_1.yaml', 'w'),default_flow_style=False)
