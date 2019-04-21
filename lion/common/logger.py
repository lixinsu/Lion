#!/usr/bin/env python
# coding: utf-8
import os
import sys
from loguru import logger as logging


def prepare_logger(log_path=None):
    # Save log to file and cleanup after 7 days
    if log_path is None:
        return logging
    if os.path.isdir(log_path):
        log_path = os.path.join(log_path, 'model.log')
    log_path = os.path.abspath(log_path)
    if not os.path.isdir(os.path.dirname(log_path)):
        raise FileNotFoundError('File not found')
    logging.add(log_path, retention="7 days")
    return logging

if __name__ == '__main__':
    prepare_logger('d.log')
    logging.info("hello")
