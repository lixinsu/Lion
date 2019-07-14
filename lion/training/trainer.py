#!/usr/bin/env python
# coding: utf-8

import sys
import math
import os.path as osp
import argparse
import json

from tensorboardX import SummaryWriter

from lion.data.loader import prepare_loader
from lion.data.dataset import LionDataset
from lion.data.vocab import Dictionary
from lion.common.param import Param
from lion.training.model import MatchingModel
from lion.common.logger import prepare_logger


MODEL_FILE = 'best_model.bin'
DEFAULTS = {'batch_size': 32,
            'epoches': 20,
            'use_cuda': False,
            'parallel': False,
            'num_workers': 2,
            'length_limit': 1000,
            'optimizer': 'adamax',
            'min_cnt': 0,
            'grad_clipping': 10,
            'weight_decay': 0,
            'embedding_file': None,
            'fix_embeddings': True,
            'sorted': True,
            'max_A_len': None,
            'max_B_len': None}


def fill_default_parameters(args):
    for k in DEFAULTS:
        if k not in args:
            args.update({k: DEFAULTS[k]})
    return args


def check_fill_parameters(args, split='train'):
    critical_keys = {'network', 'meta_dir'}
    for k in critical_keys:
        if k not in args:
            raise ValueError("Please input {} in config file".format(k))
    if split == 'train':
        if 'train_file' not in args or 'dev_file' not in args:
            raise ValueError("Train Mode must specify train_file and dev_file in config file")
    elif split == 'dev':
        if 'dev_file' not in args:
            raise ValueError("Evaluate Mode must specify 'dev_file' in config file")
    elif split == 'test':
        if 'test_file' not in args:
            raise ValueError("Predict Mode must specify 'test_file' in config file")
    return fill_default_parameters(args)


def train(output_dir):
    """Train model.

    :param output_dir: the model path which to save
    """
    config_file = osp.join(output_dir, 'params.yaml')
    args = Param.load(config_file)
    logger.info('\n' + str(args))
    args = check_fill_parameters(args, split='train')
    args.update({'output_dir': output_dir})
    writer = SummaryWriter(args.output_dir)
    for vocab_name in ['char', 'word', 'pos', 'ner', 'labelmapping']:
        if vocab_name == 'word' and 'vocab_file' in args:
            vocab_ = Dictionary.load_txt(args.vocab_file)
        else:
            vocab_ = Dictionary.load_json(osp.join(args.meta_dir, '{}.json'.format(vocab_name)), min_cnt=args.min_cnt)
        args.update({'{}_dict_size'.format(vocab_name): len(vocab_)})
        args.update({'{}_dict'.format(vocab_name): vocab_})
    args.update({'classes': len(args['labelmapping_dict'])})
    train_dataset = LionDataset(args.train_file, args)
    #TODO: num train steps
    args.num_train_optimization_steps = int(math.ceil(len(train_dataset)) / args.batch_size) * args.epoches
    dev_dataset = LionDataset(args.dev_file, args)
    train_loader = prepare_loader(train_dataset, args, split='train')
    dev_loader = prepare_loader(dev_dataset, args, split='dev')
    model = MatchingModel(args, state_dict=None)
    best_metric = 0
    for epoch in range(args.epoches):
        loss = model.train_epoch(train_loader)
        logger.info('loss {}'.format(loss))
        writer.add_scalar('train/loss', loss, epoch)
        result = model.evaluate_epoch(dev_loader)
        writer.add_scalar('dev/acc', result['acc'], epoch)
        if result['acc'] > best_metric:
            best_metric = result['acc']
            model.save(osp.join(args.output_dir, MODEL_FILE))


def evaluate(output_dir, dev_file):
    """Evaluate Model.
    :param output_dir: the model path
    :param dev_file: the dev file path
    """
    model, args = load_model(osp.join(output_dir, MODEL_FILE))
    args = check_fill_parameters(args, split='dev')
    dev_dataset = LionDataset(dev_file, args)
    dev_loader = prepare_loader(dev_dataset, args, split='dev')
    result = model.evaluate_epoch(dev_loader)
    logger.info("Acc : {}".format(result['acc']))


def predict(output_dir, test_file):
    """Predict.

    :param output_dir: the model path
    :param test_file: the test file path
    """
    model, args = load_model(osp.join(output_dir, MODEL_FILE))
    args = check_fill_parameters(args, split='test')
    test_dataset = LionDataset(test_file, args)
    test_loader = prepare_loader(test_dataset, args, split='dev')
    rv = model.predict_epoch(test_loader)
    id2label = {}
    for label, index in json.load(open(osp.join(args.meta_dir, 'labelmapping.json'))).items():
        id2label[index] = label
    for key, value in rv.items():
        rv[key] = id2label[value]
    predict_file = osp.join(output_dir, 'predictions.json')
    if osp.isfile(predict_file):
        logger.warning('Will overwrite original predictions')
    json.dump(rv, open(predict_file, 'w'))


def load_model(file_name):
    if not osp.isfile(file_name):
        raise ValueError("Model file not exit")
    return MatchingModel.load(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lion')
    parser.add_argument('--train', action='store_true', help='Training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluating')
    parser.add_argument('--predict', action='store_true', help='Predicting')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path')
    parser.add_argument('--dev_file', type=str, default='', help='File for evaluate')
    parser.add_argument('--test_file', type=str, default='', help='File for test')
    args = parser.parse_args()
    config_file = osp.join(args.output_dir, 'params.yaml')
    if not osp.isfile(config_file):
        raise ValueError("Please put a config file `params.yaml` in output_dir")
    if args.train:
        logger = prepare_logger(osp.join(args.output_dir, 'train.log'))
        logger.info('Save model in {}'.format(args.output_dir))
        train(args.output_dir)
    elif args.evaluate:
        evaluate(args.output_dir, args.dev_file)
    elif args.predict:
        predict(args.output_dir, args.test_file)
    else:
        raise ValueError("At least one of train evaluate predict shoud be true")
