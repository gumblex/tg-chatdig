#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import jieba
import pickle
import struct
import functools
import collections

def loaddict(fn):
    dic = set('、，。；？！：')
    with open(fn) as f:
        for ln in f:
            if not ln.strip():
                continue
            w = ln.split()[0]
            dic.add(w)
    return sorted(dic)

wl = loaddict(sys.argv[1])

@functools.lru_cache(maxsize=200)
def indexword(word):
    try:
        return wl.index(word)
    except ValueError:
        return None

packvals = lambda values: struct.pack('>' + 'H'*len(values), *values)

wd = collections.defaultdict(set)
for ln in sys.stdin:
    ln = set(filter(None, (indexword(word) for word in jieba.cut(ln.strip()))))
    for word in ln:
        wd[word] |= ln

pickle.dump(tuple(packvals(sorted(wd.get(k, ()))) for k in range(len(wl))), open('context.pkl', 'wb'))
