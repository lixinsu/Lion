#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from subprocess import check_call
import sys
import yaml



PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/', required=True,
                    help='Directory containing params.yaml')


def launch_training_job(parent_dir, job_name, params):
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_path = os.path.join(model_dir, 'params.yaml')
    yaml.dump(params, open(json_path, 'w'))
    cmd = "{python} run_squad.py --output_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()
    param_path = os.path.join(args.parent_dir, 'params.yaml')
    assert os.path.isfile(param_path), "No yaml configuration file found at {}".format(params_path)
    params = yaml.load(open(param_path))
    hyper_params = [True, False]
    for param in hyper_params:
        params['do_smooth'] = param
        params['debug']=False
        job_name = "T0.1_W0.5_do_smooth_{}".format(param)
        launch_training_job(args.parent_dir, job_name, params)
