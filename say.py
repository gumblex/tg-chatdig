#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import kenlm
import jieba
import pickle
import random
import itertools
import functools

RE_UCJK = re.compile(
    '([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002A6D6]+)')

punctstr = (
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~¢£¥·ˇˉ―‖‘’“”•′‵、。々'
    '〈〉《》「」『』【】〔〕〖〗〝〞︰︱︳︴︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄'
    '﹏﹐﹒﹔﹕﹖﹗﹙﹚﹛﹜﹝﹞！（），．：；？［｛｜｝～､￠￡￥')

punct = frozenset(punctstr)

def loaddict(fn):
    dic = set('、，。；？！：')
    with open(fn) as f:
        for ln in f:
            if not ln.strip():
                continue
            w = ln.split()[0]
            #if RE_UCJK.match(w):
            dic.add(w)
    return sorted(dic)


#voc = '，；。！？' + ''.join(chr(x) for x in range(0x4e00, 0x9fa5))

#end = frozenset('。！？”')


def cprint(s):
    sys.stdout.buffer.write(s.encode('utf-8'))
    sys.stdout.buffer.flush()


def weighted_choice_king(weights):
    total = 0
    winner = 0
    winweight = 0
    for i, w in enumerate(weights):
        total += w
        if random.random() * total < w:
            winner = i
            winweight = w
    return winner, winweight


def sel_best(weights):
    return max(enumerate(weights), key=lambda x: x[1])


@functools.lru_cache(maxsize=50)
def indexword(word):
    try:
        return voc.index(word)
    except ValueError:
        return None


def generate_word(lm, order, ctxvoc):
    out = []
    idx, w = weighted_choice_king(10**lm.score(c, 1, 0) for c in ctxvoc)
    sys.stdout.buffer.write(ctxvoc[idx].encode('utf-8'))
    out.append(ctxvoc[idx])
    while 1:
        bos = (len(out) <= order + 2)
        history = ' '.join(out[-order - 2:]) + ' '
        idx, w = weighted_choice_king(
            10**lm.score(history + ctxvoc[k // 2], bos, k % 2) for k in range(len(ctxvoc) * 2))
        c = ctxvoc[idx // 2]
        cprint(c)
        out.append(c)
        if idx % 2 or (len(out) > 3 and all(i == out[-1] for i in out[-3:])):
            cprint('\n')
            break


def getctxvoc(ln):
    frozenset(itertools.chain.from_iterable(map(ctx.__getitem__, filter(None, (indexword(w) for w in frozenset(jieba.cut(ln, HMM=False)))))))


LM = kenlm.LanguageModel(sys.argv[1])
order = LM.order
voc = loaddict(sys.argv[2])
ctx = pickle.load(open(sys.argv[3], 'rb'))

for ln in sys.stdin:
    ln = ln.strip()
    if ln:
        ctxvoc = list(frozenset(voc).intersection(map(voc.__getitem__, frozenset(itertools.chain.from_iterable(map(ctx.__getitem__, filter(None, map(indexword, frozenset(jieba.cut(ln, HMM=False)))))))))) or ctx
        generate_word(LM, order, ctxvoc)
    else:
        generate_word(LM, order, voc)
