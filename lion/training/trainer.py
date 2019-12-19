#!/usr/bin/env python
# coding: utf-8

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
            'length_limit': 512,
            'optimizer': 'adamax',
            'learning_rate': 0.0001,
            'min_cnt': 0,
            'grad_clipping': 10,
            'weight_decay': 0,
            'patience': 5,
            'use_elmo': None,  # 'only' or 'concat'
            'embedding_file': None,
            'fix_embeddings': True,
            'sorted': True,
            'max_A_len': None,
            'max_B_len': None}


def fill_default_parameters(params):
    for k in DEFAULTS:
        if k not in params:
            params.update({k: DEFAULTS[k]})
    return params


def check_fill_parameters(params, split='train'):
    critical_keys = {'network', 'meta_dir'}
    for k in critical_keys:
        if k not in params:
            raise ValueError("Please input {} in config file".format(k))
    if split == 'train':
        if 'train_file' not in params or 'dev_file' not in params:
            raise ValueError("Train Mode must specify train_file and dev_file in config file")
    elif split == 'dev':
        if 'dev_file' not in params:
            raise ValueError("Evaluate Mode must specify 'dev_file' in config file")
    elif split == 'test':
        if 'test_file' not in params:
            raise ValueError("Predict Mode must specify 'test_file' in config file")
    return fill_default_parameters(params)


def train():
    """Train model."""
    config_file = osp.join(args.output_dir, 'params.yaml')
    params = Param.load(config_file)
    params = check_fill_parameters(params, split='train')
    params.update({'output_dir': args.output_dir})
    writer = SummaryWriter(params.output_dir)

    for vocab_name in ['char', 'word', 'pos', 'ner', 'labelmapping']:
        if vocab_name == 'word' and 'vocab_file' in params and params['vocab_file'] is not None:
            # Load vocab from existing file
            vocab_ = Dictionary.load_vocab(params.vocab_file)
        else:
            # Load vocab from self-create file
            # Do not add <unk> and <null> for pos, ner and label dict
            if vocab_name == 'char' or vocab_name == 'word':
                add_special_tokens = True
            else:
                add_special_tokens = False
            # For label dict, min count should less than 0
            if vocab_name == 'labelmapping':
                min_count = -1
            else:
                min_count = params.min_cnt
            vocab_ = Dictionary.load_json(osp.join(params.meta_dir, '{}.json'.format(vocab_name)),
                                          min_cnt=min_count, add_special_tokens=add_special_tokens)
        params.update({'{}_dict_size'.format(vocab_name): len(vocab_)})
        params.update({'{}_dict'.format(vocab_name): vocab_})
    params.update({'classes': len(params['labelmapping_dict'])})
    logger.info('\n' + str(params))

    train_dataset = LionDataset(params.train_file, params)
    # pre-compute num train steps for `bert`
    params.num_train_optimization_steps = int(math.ceil(len(train_dataset) / params.batch_size * params.epoches))
    dev_dataset = LionDataset(params.dev_file, params)
    train_loader = prepare_loader(train_dataset, params, split='train')
    dev_loader = prepare_loader(dev_dataset, params, split='dev')

    model = MatchingModel(params, state_dict=None)
    best_metric = 0
    best_epoch = 0
    for epoch in range(params.epoches):
        loss = model.train_epoch(train_loader)
        logger.info('loss {}'.format(loss))
        writer.add_scalar('train/loss', loss, epoch)
        result = model.evaluate_epoch(dev_loader)
        writer.add_scalar('dev/acc', result['acc'], epoch)
        if result['acc'] > best_metric:
            best_epoch = epoch
            best_metric = result['acc']
            model.save(osp.join(args.output_dir, MODEL_FILE))
        elif epoch >= best_epoch+params.patience:
            break
    logger.info('Best metric:{}'.format(best_metric))


def evaluate():
    """Evaluate Model."""
    model, params = load_model(osp.join(args.output_dir, MODEL_FILE))
    params = check_fill_parameters(params, split='dev')
    dev_dataset = LionDataset(params.dev_file, params)
    dev_loader = prepare_loader(dev_dataset, params, split='dev')
    result = model.evaluate_epoch(dev_loader)
    logger.info("Acc : {}".format(result['acc']))


def predict():
    """Predict."""
    model, params = load_model(osp.join(args.output_dir, MODEL_FILE))
    params = check_fill_parameters(params, split='test')
    test_dataset = LionDataset(params.test_file, args)
    test_loader = prepare_loader(test_dataset, args, split='dev')
    rv = model.predict_epoch(test_loader)
    id2label = {}
    for label, index in json.load(open(osp.join(args.meta_dir, 'labelmapping.json'))).items():
        id2label[index] = label
    for key, value in rv.items():
        rv[key] = id2label[value]
    predict_file = osp.join(args.output_dir, 'predictions.json')
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
    args = parser.parse_args()
    config_file = osp.join(args.output_dir, 'params.yaml')
    if not osp.isfile(config_file):
        raise ValueError("Please put a config file `params.yaml` in output_dir")
    if args.train:
        logger = prepare_logger(osp.join(args.output_dir, 'train.log'))
        logger.info('Save model in {}'.format(args.output_dir))
        train()
    elif args.evaluate:
        evaluate()
    elif args.predict:
        predict()
    else:
        raise ValueError("At least one of train evaluate predict shoud be true")
