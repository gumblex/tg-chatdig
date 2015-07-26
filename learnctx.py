#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import jieba
import pickle
import collections

wd = collections.defaultdict(set)
lastline = set()
for ln in sys.stdin:
    ln = set(word for word in jieba.cut(ln.strip()))
    for word in lastline:
        wd[word] |= ln
    lastline = ln

pickle.dump(dict(wd), open('context.pkl', 'wb'))
