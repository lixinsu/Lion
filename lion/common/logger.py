#!/usr/bin/env python
# coding: utf-8
import os
import sys
from loguru import logger


def prepare_logger(log_path=None):
    logger.add(log_path, retention="7 days")
    return logger
