#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from subprocess import check_call
import sys
import copy
import yaml



PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/', required=True,
                    help='Directory containing params.yaml')
parser.add_argument('--random', action='store_true',
                    help='Random hyper-parameter search')


def launch_training_job(parent_dir, job_name, params):
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_path = os.path.join(model_dir, 'params.yaml')
    yaml.dump(params, open(json_path, 'w'), default_flow_style=False)
    cmd = "{python} lion/training/trainer.py --output_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    #check_call(cmd, shell=True)



def dfs_params(params, param_keys, param_values):
    def format_name(ks, vs):
        return '_'.join(['{}-{}'.format(k,v) for k,v in zip(ks, vs)])
    if len(param_keys) == len(param_values):
        job_name = format_name(param_keys, param_values)
        run_params = copy.copy(basic_params)
        run_params.update(dict(zip(param_keys, param_values)))
        launch_training_job(args.parent_dir, job_name, run_params)
        return
    for param in params[param_keys[len(param_values)]]:
        param_values.append(param)
        dfs_params(params, param_keys, param_values)
        param_values.pop()




if __name__ == "__main__":
    args = parser.parse_args()
    param_path = os.path.join(args.parent_dir, 'params.yaml')
    assert os.path.isfile(param_path), "No yaml configuration file found at {}".format(params_path)
    basic_params = yaml.load(open(param_path))
    tuned_param_path = os.path.join(args.parent_dir, 'tuned_params.yaml')
    assert os.path.isfile(tuned_param_path), "No tuned params configuration file"
    tuned_params = yaml.load(open(tuned_param_path))
    if not args.random:
        param_keys = list(tuned_params.keys())
        dfs_params(tuned_params, param_keys, [])

#1    param_keys = tuned_params.keys()
#1    param_values = []
#1    if args.random:
#1        for param_key in param_keys:
#1            param_values.append(random.choice(tuned_params[param_key]))
#1
