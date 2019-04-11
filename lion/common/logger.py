#!/usr/bin/env python
# coding: utf-8
import sys
from loguru import logger as logging


def prepare_logger():
    # Save log to file and cleanup after 7 days
    logging.add("model.log", retention="7 days")
    return logging

if __name__ == '__main__':
    prepare_logger()
    logging.info("hello")
