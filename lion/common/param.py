#!/usr/bin/env python
# coding: utf-8

import yaml


class Param:
    def __init__(self, dicts=None):
        self.kv = dicts or {}

    def __getitem__(self, key):
        if key not in self.kv:
            raise "Undefined parameters"
        return self.kv[key]

    def __getattr__(self, name):
        if name in self.kv:
            return self.kv[name]

    def update(self, new_kvs):
        self.kv.update(new_kvs)

    def safely_update(self, new_kvs):
        for k,v in new_kvs:
            if k not in self.kv:
                self.kv[k] = v

    @classmethod
    def load(cls, config_file):
        return cls(yaml.load(open(config_file)))

    def save(self, config_file):
        yaml.dump(self.kv, open(config_file, 'w'))


if __name__ == '__main__':
    args = Param({'bs':32, 'es':10})
