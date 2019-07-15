import os
import csv
import os.path as osp

import json
import fire


def convert_dataset(indir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for split in ['train', 'dev', 'test']:
        transform(osp.join(indir, split + '.tsv'),
                  osp.join(outdir, split + '.jsonl'), split=split)


def transform(infile, outfile, split='test'):
    result = []
    with open(infile, 'r', encoding='utf8') as input_data:
        reader = csv.reader(input_data, delimiter="\t", quotechar=None)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            if split != 'test':
                result.append({"id": line[0],
                               "A": line[1],
                               "B": line[2],
                               "label": line[3]})
            else:
                result.append({"id": line[0],
                               "A": line[1],
                               "B": line[2]})
    open(outfile, 'w').write('\n'.join([json.dumps(datum) for datum in result]))


if __name__ == '__main__':
    fire.Fire()
