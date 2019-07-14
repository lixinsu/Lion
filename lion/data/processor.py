#!/usr/bin/env python
# coding: utf-8

import os
import logging
import os.path as osp
from collections import Counter, OrderedDict

import json
from tqdm import tqdm
import fire
from multiprocessing import Pool


from lion.common.tokenizer import get_class


logger = logging.getLogger(__name__)


def gather_labels(dataset):
    label_set = set([datum['label'] for datum in dataset])
    logger.info('Total lables {}'.format(label_set))
    return {str(label):idx for idx, label in enumerate(label_set)}


def gather_dict(processed_train):
    chars = [c for datum in processed_train for token in (datum['Atokens'] + datum['Btokens']) for c in token]
    words = [token for datum in processed_train for token in (datum['Atokens'] + datum['Btokens'])]
    pos = [token for datum in processed_train for token in (datum['Apos'] + datum['Bpos'])]
    ner = [token for datum in processed_train for token in (datum['Aner'] + datum['Bner'])]
    func = lambda x:OrderedDict(Counter(x).most_common())
    return func(chars), func(words), func(pos), func(ner)


def process_datum(datum, tokenizer, label2index):
    rv = {}
    A = tokenizer.tokenize(str(datum['A']))
    B = tokenizer.tokenize(str(datum['B']))
    if 'label' in datum:
        rv['label'] = label2index[str(datum['label'])]
    rv['id'] = datum['id']
    rv['Atokens'] = A.words()
    rv['Apos'] = A.pos()
    rv['Aner'] = A.entities()
    rv['Btokens'] = B.words()
    rv['Bpos'] = B.pos()
    rv['Bner'] = B.entities()
    op = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if callable(op):
        # Adapt to bert input format
        rv['Atokens'] = ["[CLS]"] + rv['Atokens'] + ["[SEP]"]
        rv['A_ids'] = tokenizer.convert_tokens_to_ids(rv['Atokens'])
        rv['Btokens'] = rv['Btokens'] + ["[SEP]"]
        rv['B_ids'] = tokenizer.convert_tokens_to_ids(rv['Btokens'])
    return rv


def process_dataset(in_dir, out_dir, tokenizer_name='spacy', vocab_file=None, splits=['train', 'dev', 'test']):

    def jsondump(data, filename):
        json.dump(data, open(osp.join(out_dir, filename), 'w'), indent=2)
    if tokenizer_name == 'bert':
        tokenizer = get_class(tokenizer_name)(vocab_file)
    else:
        tokenizer = get_class(tokenizer_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    if 'train' in splits:
        split = 'train.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename)]
        label2index = gather_labels(dataset)
        jsondump(label2index, 'labelmapping.json')
        processed = []
        for datum in tqdm(dataset):
            try:
                processed.append(process_datum(datum, tokenizer, label2index))
            except:
                logger.info('Bae line {}'.format(datum))
        #with Pool(30) as p:
        #    processed = p.map(tokenizer.tokenize, dataset)
        char_dict, word_dict, pos_dict, ner_dict = gather_dict(processed)
        jsondump(char_dict, 'char.json')
        jsondump(word_dict, 'word.json')
        jsondump(pos_dict, 'pos.json')
        jsondump(ner_dict, 'ner.json')
        out_file = open(osp.join(out_dir, 'train_{}.jsonl'.format(tokenizer_name)), 'w')
        for datum in processed:
            out_file.write('{}\n'.format(json.dumps(datum)))
    if 'dev' in splits:
        split = 'dev.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename)]
        out_file = open(osp.join(out_dir, 'dev_{}.jsonl'.format(tokenizer_name)), 'w')
        for datum in tqdm(dataset):
            processed_datum = process_datum(datum, tokenizer, label2index)
            out_file.write('{}\n'.format(json.dumps(processed_datum)))
    if 'test' in splits:
        split = 'test.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename)]
        out_file = open(osp.join(out_dir, 'test_{}.jsonl'.format(tokenizer_name)), 'w')
        for datum in tqdm(dataset):
            processed_datum = process_datum(datum, tokenizer, label2index)
            out_file.write('{}\n'.format(json.dumps(processed_datum)))


if __name__ == '__main__':
    fire.Fire()

