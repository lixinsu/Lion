#!/usr/bin/env python
# coding: utf-8

import json

import torch
from torch.utils.data.dataset import Dataset
from loguru import logger


class LionDataset(Dataset):

    def __init__(self, data_file, args):
        self.args = args
        self.pos_dict = args.pos_dict
        self.ner_dict = args.ner_dict
        self.word_dict = args.word_dict
        self.char_dict = args.char_dict
        self.length_limit = args.length_limit
        self.examples = self._load_json(data_file)

    def _load_json(self, data_file):
        data = [json.loads(line) for line in open(data_file)]
        ori = len(data)
        data = [d for d in data if len(d['Atokens'])+len(d['Btokens']) <= self.args.length_limit]
        logger.info('{} filter {} abnormal instance'.format(data_file, ori - len(data)))
        return data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.vectorize(self.examples[index])

    def get_origin(self, index):
        return self.examples[index]

    def vectorize(self, ex):
        def make_char(char_dict, token, word_length=16):
            if len(token) > 16:
                return [char_dict[t_] for t_ in token[:8]] + [char_dict[t_] for t_ in token[-8:]]
            else:
                rv = [0] * 16
                for i in range(len(token)):
                    rv[i] = char_dict[token[i]]
                return rv

        # Index words
        word_dict = self.word_dict
        char_dict = self.char_dict
        pos_dict = self.pos_dict
        ner_dict = self.ner_dict
        if self.length_limit and len(ex['Atokens']) > self.length_limit:
            ex['Atokens'] = ex['Atokens'][0:self.length_limit]
            ex['Apos'] = ex['Apos'][0:self.length_limit]
            ex['Aner'] = ex['Aner'][0:self.length_limit]
        if self.length_limit and len(ex['Btokens']) > self.length_limit:
            ex['Btokens'] = ex['Btokens'][0:self.length_limit]
            ex['Bpos'] = ex['Bpos'][0:self.length_limit]
            ex['Bner'] = ex['Bner'][0:self.length_limit]
        oriAtoken, oriBtoken = None, None
        if self.args.use_elmo:
            # Not index words
            oriAtoken = ex['Atokens']
            oriBtoken = ex['Btokens']
        if self.args.network == 'xlnet':
            Atoken = torch.LongTensor(ex['Atokens'])
            Btoken = torch.LongTensor(ex['Btokens'])
            Achar = torch.zeros(len(ex['Atokens']), 16)
            Bchar = torch.zeros(len(ex['Btokens']), 16)
        else:
            Atoken = torch.LongTensor([word_dict[w] for w in ex['Atokens']])
            Btoken = torch.LongTensor([word_dict[w] for w in ex['Btokens']])
            Achar = torch.LongTensor([make_char(char_dict, w) for w in ex['Atokens']])
            Bchar = torch.LongTensor([make_char(char_dict, w) for w in ex['Btokens']])

        Apos = torch.LongTensor([pos_dict[w] if w is not None else 0 for w in ex['Apos']])
        Bpos = torch.LongTensor([pos_dict[w] if w is not None else 0 for w in ex['Bpos']])

        Aner = torch.LongTensor([ner_dict[w] if w is not None else 0 for w in ex['Aner']])
        Bner = torch.LongTensor([ner_dict[w] if w is not None else 0 for w in ex['Bner']])

        rv = {'id': ex['id'],
              'Atoken': oriAtoken,
              'Btoken': oriBtoken,
              'Atoken_ids': Atoken,
              'Btoken_ids': Btoken,
              'Achar_ids': Achar,
              'Bchar_ids': Bchar,
              'Apos_ids': Apos,
              'Bpos_ids': Bpos,
              'Aner_ids': Aner,
              'Bner_ids': Bner}
        if 'label' in ex:
            rv['label'] = ex['label']
        return rv

    def lengths(self):
        return [(len(ex['Atokens']), len(ex['Btokens'])) for ex in self.examples]


if __name__ == '__main__':
    dataset = LionDataset('data/preprocessed/QQPdebug/train_spacy.jsonl', 'data/preprocessed/QQPdebug/')
