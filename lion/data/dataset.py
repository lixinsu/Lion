#!/usr/bin/env python
# coding: utf-8

import json
import os.path as osp

import torch
from torch.utils.data.dataset import Dataset

from lion.data.vocab import Dictionary



class LionDataset(Dataset):

    def __init__(self, filename, meta_dir):
        self.examples = [json.loads(line) for line in open(filename)]
        self.word_dict = Dictionary.load(osp.join(meta_dir, 'word.json'))
        self.char_dict = Dictionary.load(osp.join(meta_dir, 'char.json'))
        self.pos_dict =  Dictionary.load(osp.join(meta_dir, 'pos.json'))
        self.ner_dict = Dictionary.load(osp.join(meta_dir, 'ner.json'))

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
        Atoken = torch.LongTensor([word_dict[w] for w in ex['Atokens']])
        Btoken = torch.LongTensor([word_dict[w] for w in ex['Btokens']])

        Apos = torch.LongTensor([pos_dict[w] for w in ex['Apos']])
        Bpos = torch.LongTensor([pos_dict[w] for w in ex['Bpos']])

        Aner = torch.LongTensor([ner_dict[w] for w in ex['Aner']])
        Bner = torch.LongTensor([ner_dict[w] for w in ex['Bner']])

        def make_char(char_dict, token, word_length=16):
            if len(token) > 16:
                return [char_dict(t_) for t_ in token[:8]] + [char_dict(t_) for t_ in token[-8:]]
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

