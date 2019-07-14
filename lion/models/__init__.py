#!/usr/bin/env python
# coding: utf-8


from .test_model_1 import TestModel_1
from .bimpm import BIMPM
from .esim import ESIM
from .bert import BertForSequenceClassification


def get_model_class(name):
    if name == 'test_model_1':
        return TestModel_1
    if name == 'bimpm':
        return BIMPM
    if name == 'esim':
        return ESIM
    if name == 'bert':
        return BertForSequenceClassification
    raise RuntimeError('Invalid model %s' % name)
