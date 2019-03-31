#!/usr/bin/env python
# coding: utf-8


import os.path as osp

from lion.data.loader import prepare_loader
from lion.data.dataset import LionDataset
from lion.data.vocab import Dictionary
from lion.common.param import Param
from lion.training.model import MatchingModel


def train_model(config_file):
    args = Param.load(config_file)
    for vocab_name in ['char', 'word', 'pos', 'ner']:
        vocab_ = Dictionary.load(osp.join(args.meta_dir, '{}.json'.format(vocab_name)), min_cnt=args.min_cnt)
        args.update({'{}_dict_size'.format(vocab_name): len(vocab_)})
    train_dataset = LionDataset(args.train_file, args.meta_dir)
    dev_dataset = LionDataset(args.dev_file, args.meta_dir)
    train_loader = prepare_loader(train_dataset, args, split='train')
    dev_loader = prepare_loader(dev_dataset, args, split='dev')
    model = MatchingModel(args, state_dict=None)
    for epoch in range(args.epoches):
        model.train_epoch(train_loader)
        res = model.predict_epoch(dev_loader)
        if args.visualization:
            pass


if __name__ == '__main__':
    # train_model('lion/configs/test_model_1.yaml')
    train_model('lion/configs/test_bimpm_1.yaml')
    # train_model('lion/configs/test_esim_1.yaml')


