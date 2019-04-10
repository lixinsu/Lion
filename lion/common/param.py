#!/usr/bin/env python
# coding: utf-8

import yaml


class Param:
    def __init__(self, dicts=None):
        self.kv = dicts or {}

    def __getitem__(self, key):
        if key not in self.kv:
            raise ValueError("Undefined parameters {}".format(key))
        return self.kv[key]

    def __getattr__(self, key):
        if key in self.kv:
            return self.kv[key]
        raise ValueError("Undefined parameters {}".format(key))

    def __contains__(self, key):
        return True if key in self.kv else False

    def update(self, new_kvs):
        self.kv.update(new_kvs)

    def safely_update(self, new_kvs):
        for k,v in new_kvs:
            if k not in self.kv:
                self.kv[k] = v

    def __getstate__(self):
        return self.kv

    def __setstate__(self, d):
        self.kv = d

    @classmethod
    def load(cls, config_file):
        return cls(yaml.load(open(config_file)))

    def save(self, config_file):
        yaml.dump(self.kv, open(config_file, 'w'))


if __name__ == '__main__':
    args = Param({'bs':32, 'es':10})
