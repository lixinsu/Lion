#!/usr/bin/env python
# coding: utf-8

import yaml


rv = {}

# Path
rv['meta_dir'] = 'data/preprocessed/QQPdebug/'
rv['train_file'] = 'data/preprocessed/QQPdebug/train_spacy.jsonl'
rv['dev_file'] = 'data/preprocessed/QQPdebug/train_spacy.jsonl'
rv['test_file'] = 'data/preprocessed/QQPdebug/test_spacy.jsonl'
rv['result_dir'] = 'data/outputs/QQPdebug/'
# rv['embedding_file'] = ''
rv['embedding_file'] = '/home/fanyixing/users/mxy/coqa-baselines/wordvecs/glove.42B.300d.txt'

# General
rv['batch_size'] = 64
rv['epoches'] = 20
rv['num_workers'] = 2
rv['use_cuda'] = True

# Model
rv['network'] = 'esim'
rv['min_cnt'] = 0
rv['word_dim'] = 300
rv['hidden_size'] = 100
rv['classes'] = 2
rv['dropout'] = 0.1
rv['optimizer'] = 'adamax'
rv['weight_decay'] = 0
rv['grad_clipping'] = 10


#rv['max_seq_length'] = 50
#rv['max_word_length'] = 16
#rv['num_perspective'] = 20
#rv['use_char_emb'] = True
#rv['char_hidden_size'] = 50
#rv['char_dim'] = 20
#rv['rnn_layers'] = 2
#rv['fix_embeddings'] = False
#
# Vocab信息后更新
yaml.dump(rv, open('lion/configs/esim.yaml', 'w'),default_flow_style=False)
