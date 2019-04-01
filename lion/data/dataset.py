#!/usr/bin/env python
# coding: utf-8

import json
import os.path as osp

import torch
from torch.utils.data.dataset import Dataset

from lion.data.vocab import Dictionary


class LionDataset(Dataset):

    def __init__(self, args, split='train'):
        if split == 'train':
            self.examples = [json.loads(line) for line in open(args.train_file)]
        elif split == 'dev':
            self.examples = [json.loads(line) for line in open(args.dev_file)]
        elif split == 'test':
            self.examples = [json.loads(line) for line in open(args.test_file)]
        else:
            raise ValueError("split must be set with train, dev or test!")
        self.length_limit = args.length_limit
        self.word_dict = Dictionary.load(osp.join(args.meta_dir, 'word.json'))
        self.char_dict = Dictionary.load(osp.join(args.meta_dir, 'char.json'))
        self.pos_dict = Dictionary.load(osp.join(args.meta_dir, 'pos.json'))
        self.ner_dict = Dictionary.load(osp.join(args.meta_dir, 'ner.json'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.vectorize(self.examples[index])

    def get_origin(self, index):
        return self.examples[index]

    def vectorize(self, ex):
        # Index words
        word_dict = self.word_dict
        char_dict = self.char_dict
        pos_dict = self.pos_dict
        ner_dict = self.ner_dict
        if not self.length_limit and len(ex['Atokens']) > self.length_limit:
            ex['Atokens'] = ex['Atokens'][0:self.length_limit]
        if not self.length_limit and len(ex['Btokens']) > self.length_limit:
            ex['Btokens'] = ex['Btokens'][0:self.length_limit]
        Atoken = torch.LongTensor([word_dict[w] for w in ex['Atokens']])
        Btoken = torch.LongTensor([word_dict[w] for w in ex['Btokens']])

        Apos = torch.LongTensor([pos_dict[w] for w in ex['Apos']])
        Bpos = torch.LongTensor([pos_dict[w] for w in ex['Bpos']])

        Aner = torch.LongTensor([ner_dict[w] for w in ex['Aner']])
        Bner = torch.LongTensor([ner_dict[w] for w in ex['Bner']])

        def make_char(char_dict, token, word_length=16):
            if len(token) > 16:
                return [char_dict[t_] for t_ in token[:8]] + [char_dict[t_] for t_ in token[-8:]]
            else:
                rv = [0] * 16
                for i in range(len(token)):
                    rv[i] = char_dict[token[i]]
                return rv

        Achar = torch.LongTensor([make_char(char_dict, w) for w in ex['Atokens']])
        Bchar = torch.LongTensor([make_char(char_dict, w) for w in ex['Btokens']])

        rv = {  'id': ex['id'],
                'Atoken': Atoken,
                'Btoken': Btoken,
                'Achar': Achar,
                'Bchar': Bchar,
                'Apos': Apos,
                'Bpos': Bpos,
                'Aner': Aner,
                'Bner': Bner}
        if 'label' in ex:
            rv['label'] = ex['label']
        return rv

    def lengths(self):
        return [(len(ex['Atokens']), len(ex['Btokens']))
                                for ex in self.examples]


if __name__ == '__main__':
    dataset = LionDataset('data/preprocessed/QQPdebug/train_spacy.jsonl', 'data/preprocessed/QQPdebug/')


