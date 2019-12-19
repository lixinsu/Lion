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

    def __setitem__(self, key, value):
        self.kv[key] = value

    def __setattr__(self, name, value):
        if name == 'kv':
            if hasattr(self, 'kv'):
                raise ValueError("kv is reserved keyword for Param class")
            super().__setattr__(name, value)
        else:
            self.kv[name] = value

    def __getattribute__(self, key):
        if key == '__dict__':
            return self.kv
        return super().__getattribute__(key)

    def __getattr__(self, key):
        if key == 'kv':
            super().__getattr__()
        if key in self.kv:
            return self.kv[key]
        raise ValueError("Undefined parameters {}".format(key))

    def __getstate__(self):
        return self.kv

    def __setstate__(self, data):
        self.kv = data

    def __contains__(self, key):
        return True if key in self.kv else False

    def update(self, new_kvs):
        self.kv.update(new_kvs)

    def safely_update(self, new_kvs):
        for k,v in new_kvs:
            if k not in self.kv:
                self.kv[k] = v

    def __str__(self):
        return '\n'.join(['{} = {}'.format(k,v) for k,v in self.kv.items()])

    @classmethod
    def load(cls, config_file):
        return cls(dicts=yaml.load(open(config_file)))

    def save(self, config_file):
        yaml.dump(self.kv, open(config_file, 'w'))
