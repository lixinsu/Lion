#!/usr/bin/env python
# coding: utf-8


import json
import unicodedata


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self, add_special_tokens=True):
        self.tok2ind, self.ind2tok = {}, {}
        if add_special_tokens:
            self.tok2ind = {self.NULL: 0, self.UNK: 1}
            self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens

    @classmethod
    def load_json(cls, vocab_file, min_cnt=0, add_special_tokens=True):
        """Loads our preprocessed vocab json file."""
        vocab = cls(add_special_tokens)
        for k, v in json.load(open(vocab_file)).items():
            assert v is not None
            if v > min_cnt:
                vocab.add(k)
        return vocab

    @classmethod
    def load_vocab(cls, vocab_file):
        """Loads an existing vocabulary file ."""
        vocab = cls(add_special_tokens=False)
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

if __name__ == '__main__':
    vocab = Dictionary.load('data/preprocessed/QQPdebug/ner.json')
