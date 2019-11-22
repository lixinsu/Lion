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


def batchify_factory(max_A_len=None, max_B_len=None, elmo_batch=None, network=None):
    def batchify(batch):
        key_list = ['Atoken_ids', 'Apos_ids', 'Aner_ids', 'Achar_ids',
                    'Btoken_ids', 'Bpos_ids', 'Bner_ids', 'Bchar_ids']
        ids = [ex['id'] for ex in batch]
        labels = [ex.get('label', 0) for ex in batch]
        rv = {}
        rv['ids'] = ids
        rv['labels'] = torch.LongTensor(labels)
        Amask, Bmask, Asegment, Bsegment = None, None, None, None
        if elmo_batch:
            key_list += ['Atoken', 'Btoken']
        for k in key_list:
            batch_data = [ex[k] for ex in batch]
            if k.endswith('token') and elmo_batch:
                batch_data = elmo_batch(batch_data)
            unified_max_len = max_A_len if 'A' in k else max_B_len  # For CNN model with fixed length on whole dataset
            current_max_len = max([d.size(0) for d in batch_data])
            max_len = unified_max_len or current_max_len

            if 'char' not in k:
                if k.endswith('token') and elmo_batch:
                    # 50 is the character dim of elmo
                    padded_data = torch.LongTensor(len(batch_data), max_len, 50).fill_(0)
                else:
                    padded_data = torch.LongTensor(len(batch_data), max_len).fill_(0)
            else:
                padded_data = torch.LongTensor(len(batch_data), max_len, 16).fill_(0)

            if 'A' in k and 'Amask' not in rv and 'Asegment' not in rv:
                # The mask has 1 for real tokens and 0 for padding tokens, this is how bert does!
                # So tranditional model like esim need to inverse it.
                Amask = torch.LongTensor(len(batch_data), max_len).fill_(0)
                Asegment = torch.LongTensor(len(batch_data), max_len).fill_(0)
            if 'B' in k and 'Bmask' not in rv and 'Bsegment' not in rv:
                Bmask = torch.LongTensor(len(batch_data), max_len).fill_(0)
                Bsegment = torch.LongTensor(len(batch_data), max_len).fill_(1)
                if network == 'xlnet' and Bsegment is not None:
                    # for <cls>, segment id is 2
                    for i, d in enumerate(batch_data):
                        Bsegment[i, d.size(0)-1].fill_(2)

            for i, d in enumerate(batch_data):
                if 'char' not in k:
                    padded_data[i, :d.size(0)].copy_(d[:max_len])
                else:
                    padded_data[i, :d.size(0), :].copy_(d[:max_len, :])
                if Amask is not None:
                    Amask[i, :d.size(0)].fill_(1)
                if Bmask is not None:
                    Bmask[i, :d.size(0)].fill_(1)

            if 'Amask' not in rv and Amask is not None:
                rv['Amask'] = Amask
                rv['Asegment'] = Asegment
            if 'Bmask' not in rv and Bmask is not None:
                rv['Bmask'] = Bmask
                rv['Bsegment'] = Bsegment
            rv[k] = padded_data
        return rv
    return batchify


def prepare_loader(dataset, args, split='train'):
    batch_to_ids = None
    if args.use_elmo:
        from allennlp.modules.elmo import batch_to_ids
    if args.sorted and split == 'train':
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
        collate_fn=batchify_factory(args.max_A_len, args.max_B_len, batch_to_ids, args.network),
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
