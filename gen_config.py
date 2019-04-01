#!/usr/bin/env python
# coding: utf-8

import yaml


rv = {}

# Path
rv['meta_dir'] = 'data/preprocessed/QQPdebug/'
rv['train_file'] = 'data/preprocessed/QQPdebug/train_spacy.jsonl'
rv['dev_file'] = 'data/preprocessed/QQPdebug/dev_spacy.jsonl'
rv['test_file'] = 'data/preprocessed/QQPdebug/test_spacy.jsonl'
rv['result_dir'] = 'data/outputs/QQPdebug/'
rv['embedding_file'] = ''


# General
rv['batch_size'] = 32
rv['epoches'] = 10
rv['num_workers'] = 2
rv['use_cuda'] = True

# Model
rv['network'] = 'test_model_1'
rv['hidden_size'] = 64
rv['rnn_layers'] = 2
rv['classes'] = 2
rv['word_dim'] = 300
rv['min_cnt'] = 0
rv['fix_embeddings'] = False
rv['optimizer'] = 'adamax'
rv['weight_decay'] = 0
rv['grad_clipping'] = 10

# Vocab信息后更新


yaml.dump(rv, open('lion/configs/test_model_1.yaml', 'w'),default_flow_style=False)
