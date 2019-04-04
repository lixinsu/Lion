#!/usr/bin/env python
# coding: utf-8

import sys
import os.path as osp

from lion.data.loader import prepare_loader
from lion.data.dataset import LionDataset
from lion.data.vocab import Dictionary
from lion.common.param import Param
from lion.training.model import MatchingModel
from lion.common.logger import prepare_logger


def train_model(config_file):
    args = Param.load(config_file)
    for vocab_name in ['char', 'word', 'pos', 'ner', 'labelmapping']:
        vocab_ = Dictionary.load(osp.join(args.meta_dir, '{}.json'.format(vocab_name)), min_cnt=args.min_cnt)
        args.update({'{}_dict_size'.format(vocab_name): len(vocab_)})
        args.update({'{}_dict'.format(vocab_name): vocab_})
    args.update({'classes': len(args['labelmapping_dict'])})
    train_dataset = LionDataset(args, split='train')
    dev_dataset = LionDataset(args, split='dev')
    train_loader = prepare_loader(train_dataset, args, split='train')
    dev_loader = prepare_loader(dev_dataset, args, split='dev')
    model = MatchingModel(args, state_dict=None)
    for epoch in range(args.epoches):
        model.train_epoch(train_loader)
        model.evaluate_epoch(dev_loader)
        if args.visualization:
            pass
    model.save(osp.join(args['result_dir'], 'pytorch_model.bin'))


def evaluate(config_file):
    args = Param.load(config_file)
    model = load_model(osp.join(args['result_dir'], 'pytorch_model.bin'))
    dev_dataset = LionDataset(args, split='dev')
    dev_loader = prepare_loader(dev_dataset, args, split='dev')
    model.evaluate_epoch(dev_loader)


def predict(config_file):
    args = Param.load(config_file)
    model = load_model(osp.join(args['result_dir'], 'pytorch_model.bin'))
    test_dataset = LionDataset(args, split='dev')
    test_loader = prepare_loader(test_dataset, args, split='dev')
    model.predict_epoch(test_loader)


def load_model(file_name):
    model = MatchingModel.load(file_name)
    return model


if __name__ == '__main__':
    train_model(sys.argv[1])
