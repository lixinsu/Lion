#!/usr/bin/env python
# coding: utf-8

import json

import numpy as np
import torch
from torch.utils import data

from lion.data.dataset import LionDataset


class SortedBatchSampler(data.Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)


def batchify_factory(max_A_len=None, max_B_len=None):
    if (not max_A_len) and (not max_B_len):
        max_A_len, max_B_len = 256, 256
    def batchify(batch):
        ids = [ex['id'] for ex in batch]
        labels = [ex.get('label', 0) for ex in batch]
        rv = {}
        rv['ids'] = ids
        rv['labels'] = torch.LongTensor(labels)
        Amask, Bmask = None, None

        for k in ['Atoken', 'Apos', 'Aner', 'Btoken', 'Bpos', 'Bner', 'Achar', 'Bchar']:
            batch_data = [ex[k] for ex in batch]
            unified_max_len = max_A_len if 'A' in k else max_B_len      # For CNN model with fixed length on whole dataset
            current_max_len = max([d.size(0) for d in batch_data])
            max_len = unified_max_len if current_max_len > unified_max_len else current_max_len
            if 'char' not in k:
                padded_data = torch.LongTensor(len(batch_data), max_len).fill_(0)
            else:
                padded_data = torch.LongTensor(len(batch_data), max_len, 16).fill_(0)
            if 'A' in k and 'Amask' not in rv:
                Amask = torch.ByteTensor(len(batch_data), max_len).fill_(1)
            if 'B' in k and 'Bmask' not in rv:
                Bmask = torch.ByteTensor(len(batch_data), max_len).fill_(1)
            for i, d in enumerate(batch_data):
                if 'char' not in k:
                    padded_data[i, :d.size(0)].copy_(d[:max_len])
                else:
                    padded_data[i, :d.size(0),:].copy_(d[:max_len, :])
                if Amask is not None:
                    Amask[i,:d.size(0)].fill_(0)
                if Bmask is not None:
                    Bmask[i, :d.size(0)].fill_(0)
            if 'Amask' not in rv and Amask is not None:
                rv['Amask'] = Amask
            if 'Bmask' not in rv and Bmask is not None:
                rv['Bmask'] = Bmask
            rv[k] = padded_data
        return rv
    return batchify


def prepare_loader(dataset, args, split='train'):
    if args.sorted and split=='train':
        sampler = SortedBatchSampler(dataset.lengths(), args.batch_size, shuffle=True)
    elif split == 'train':
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.SequentialSampler(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=batchify_factory(args.max_A_len, args.max_B_len),
        pin_memory=True,
    )
    return loader


if __name__ == '__main__':
    dataset = LionDataset('data/preprocessed/QQPdebug/train_spacy.jsonl', 'data/preprocessed/QQPdebug/')
    class foo():
        pass
    args = foo()
    args.batch_size = 4
    args.sorted = True
    args.data_workers = 4
    args.max_A_len = None
    args.max_B_len = None
    train_loader = prepare_loader(dataset, args, split='train')

