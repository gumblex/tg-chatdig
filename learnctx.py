#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import jieba
import pickle
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

wd = collections.defaultdict(set)
lastline = set()
for ln in sys.stdin:
    ln = set(filter(None, (indexword(word) for word in jieba.cut(ln.strip()))))
    for word in lastline:
        wd[word] |= ln
    lastline = ln

pickle.dump(tuple(tuple(sorted(wd.get(k, ()))) for k in range(len(wl))), open('context.pkl', 'wb'))
