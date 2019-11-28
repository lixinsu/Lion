#!/usr/bin/env python
# coding: utf-8

import os
import json
import fire
import logging
from tqdm import tqdm
import os.path as osp
from collections import Counter, OrderedDict

from lion.common.tokenizer import get_class

logger = logging.getLogger(__name__)


def gather_labels(dataset):
    label_set = set([datum['label'] for datum in dataset])
    logger.info('Total lables {}'.format(label_set))
    return {str(label): idx for idx, label in enumerate(label_set)}


def gather_dict(processed_train):
    chars = [c for datum in processed_train for token in (datum['Atokens'] + datum['Btokens']) for c in token]
    words = [token for datum in processed_train for token in (datum['Atokens'] + datum['Btokens'])]
    pos = [token for datum in processed_train for token in (datum['Apos'] + datum['Bpos'])]
    ner = [token for datum in processed_train for token in (datum['Aner'] + datum['Bner'])]
    func = lambda x: OrderedDict(Counter(x).most_common())
    return func(chars), func(words), func(pos), func(ner)


def process_datum(datum, tokenizer, label2index, max_length):
    assert len(datum) <= 4
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
    if tokenizer.__class__.__name__ == 'BertTokenizer':
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A Bert sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        truncate_seq_pair(rv['Atokens'], rv['Btokens'], max_length-3)
        rv['Atokens'] = ["[CLS]"] + rv['Atokens'] + ["[SEP]"]
        rv['Btokens'] = rv['Btokens'] + ["[SEP]"]
    if tokenizer.__class__.__name__ == 'XLNetTokenizer':
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A xlnet sequence pair has the following format: A [SEP] B [SEP] [CLS]
        special_symbols = {SEG_ID_A: 0, SEG_ID_B: 1, SEG_ID_CLS: 2, "<cls>": 3, "<sep>": 4,
        SEG_ID_SEP: 3, SEG_ID_PAD: 4}
        """
        truncate_seq_pair(rv['Atokens'], rv['Btokens'], max_length-3)
        rv['Atokens'] = rv['Atokens'] + [4]
        rv['Btokens'] = rv['Btokens'] + [4] + [3]
    if tokenizer.__class__.__name__ == 'RobertaTokenizer':
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s></s> B </s>
        """
        truncate_seq_pair(rv['Atokens'], rv['Btokens'], max_length - 3)
        rv['Atokens'] = [0] + rv['Atokens'] + [2] + [2]
        rv['Btokens'] = rv['Btokens'] + [2]
    return rv


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_dataset(in_dir, out_dir, splits=['train', 'dev', 'test'],
                    tokenizer_name='spacy', vocab_file=None, max_length=128, **kwargs):
    def jsondump(data, filename):
        json.dump(data, open(osp.join(out_dir, filename), 'w'), indent=2, ensure_ascii=False)

    if tokenizer_name == 'bert' or tokenizer_name == 'xlnet':
        if not vocab_file:
            raise ValueError('Bert model should use an existing vocab')
        tokenizer = get_class(tokenizer_name)(vocab_file)
    elif tokenizer_name == 'roberta':
        merges_file = kwargs.pop('merges_file')
        tokenizer = get_class(tokenizer_name)(vocab_file, merges_file)
    else:
        tokenizer = get_class(tokenizer_name)()

    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if 'train' in splits:
        split = 'train.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename, 'r', encoding='utf-8').readlines()]
        label2index = gather_labels(dataset)
        jsondump(label2index, 'labelmapping.json')
        processed = []
        for datum in tqdm(dataset):
            try:
                processed.append(process_datum(datum, tokenizer, label2index, max_length))
            except Exception as e:
                raise ValueError('Exception: {}, Bae line {}'.format(e, datum, ensure_ascii=False))
        if tokenizer_name != 'xlnet' and tokenizer_name != 'roberta':
            char_dict, word_dict, pos_dict, ner_dict = gather_dict(processed)
            jsondump(char_dict, 'char.json')
            jsondump(word_dict, 'word.json')
            jsondump(pos_dict, 'pos.json')
            jsondump(ner_dict, 'ner.json')
        out_file = open(osp.join(out_dir, 'train_{}.jsonl'.format(tokenizer_name)), 'w')
        for datum in processed:
            out_file.write('{}\n'.format(json.dumps(datum, ensure_ascii=False)))

    if 'train' not in splits:
        raise ValueError('`splits` argument must contain `train` otherwise `label2index` will be NontType!')

    if 'dev' in splits:
        split = 'dev.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename).readlines()]
        out_file = open(osp.join(out_dir, 'dev_{}.jsonl'.format(tokenizer_name)), 'w')
        processed = []
        for datum in tqdm(dataset):
            try:
                processed.append(process_datum(datum, tokenizer, label2index, max_length))
            except Exception as e:
                raise ValueError('Exception: {}, Bae line {}'.format(e, datum, ensure_ascii=False))
        for datum in processed:
            out_file.write('{}\n'.format(json.dumps(datum)))
    if 'test' in splits:
        split = 'test.jsonl'
        filename = osp.join(in_dir, split)
        dataset = [json.loads(line) for line in open(filename).readlines()]
        out_file = open(osp.join(out_dir, 'test_{}.jsonl'.format(tokenizer_name)), 'w')
        processed = []
        for datum in tqdm(dataset):
            try:
                processed.append(process_datum(datum, tokenizer, label2index, max_length))
            except Exception as e:
                raise ValueError('Exception: {}, Bae line {}'.format(e, datum, ensure_ascii=False))
        for datum in processed:
            out_file.write('{}\n'.format(json.dumps(datum, ensure_ascii=False)))


if __name__ == '__main__':
    fire.Fire()
