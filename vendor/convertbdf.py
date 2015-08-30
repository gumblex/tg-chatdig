#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import bdflib

def packrow(iterable):
    v = 0
    for bit in iterable:
        v = (v<<1) | bit
    return v

def loadfrombdf(filename):
    srcfile = open(filename, 'r')
    fontd = bdflib.read_bdf(srcfile)
    srcfile.close()
    glyphs = {}
    for k, v in fontd.glyphs_by_codepoint.items():
        llen = len(v.bitmap()[0])
        glyphs[k] = (llen,) + tuple(packrow(l) for l in v.bitmap())
    maxnum = max(glyphs)
    return tuple(glyphs.get(g) for g in range(max(glyphs)+1))

glyphs = loadfrombdf(sys.argv[1])
pickle.dump(glyphs, open(sys.argv[2], 'wb'))
