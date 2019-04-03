import os
import os.path as osp

import json
import fire


def convert_dataset(indir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for split in ['train', 'dev', 'test']:
        transform(osp.join(indir, 'snli_1.0_' + split + '.txt'),\
                osp.join(outdir, split + '.jsonl'), split=split)


def transform(infile, outfile, split='test'):
    with open(infile, 'r', encoding='utf8') as input_data:
        # Translation tables to remove parentheses and punctuation from
        # strings.
        parentheses_table = str.maketrans({'(': None, ')': None})
        # Ignore the headers on the first line of the file.
        next(input_data)
        result = []
        for line in input_data:
            line = line.strip().split('\t')
            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue
            pair_id = line[7]
            premise = line[1]
            hypothesis = line[2]
            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            premise = ' '.join([w for w in premise.rstrip().split()])
            hypothesis = hypothesis.translate(parentheses_table)
            hypothesis = ' '.join([w for w in hypothesis.rstrip().split()])
            label = line[0]
            id = pair_id
            if split != 'test':
                result.append({"id": id,
                               "A": premise,
                               "B": hypothesis,
                               "label": label})
            else:
                result.append({"id": id,
                               "A": premise,
                               "B": hypothesis})
    open(outfile, 'w').write('\n'.join([json.dumps(datum) for datum in result]))


if __name__ == '__main__':
    fire.Fire()
