#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp

import json
import fire
import numpy as np
import pandas as pd


def convert_dataset(indir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for split in ['train', 'dev', 'test']:
        transform(osp.join(indir, split + '.tsv'), \
                osp.join(outdir, split + '.jsonl'), split=split)


def transform(infile, outfile, split='test'):
    df = pd.read_csv(infile, sep='\t', error_bad_lines=False)
    if split != 'test':
        sub_df = df[['id', 'question1', 'question2', 'is_duplicate']]
        sub_df.columns = ['id', 'A', 'B', 'label']
    else:
        sub_df = df[['id', 'question1', 'question2']]
        sub_df.columns = ['id', 'A', 'B']
    open(outfile, 'w').write('\n'.join([json.dumps(datum) for datum in sub_df.to_dict(orient='records')]))


if __name__ == '__main__':
    fire.Fire()





