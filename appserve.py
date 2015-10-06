#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import queue
import tempfile
import threading
import traceback
import subprocess
import collections

from vendor import zhutil
from vendor import zhconv
from vendor import figchar
from vendor import simpleime
from vendor import mosesproxy
from vendor import chinesename
#from vendor import fparser

# {"id": 1, "cmd": "bf", "args": [",[.,]", "asdasdf"]}

def docommands():
    global MSG_Q
    while 1:
        obj = MSG_Q.get()
        thr = threading.Thread(target=async_command, args=(obj,))
        thr.daemon = True
        thr.start()

def async_command(obj):
    sys.stdout.buffer.write(json.dumps(process(obj)).encode('utf-8') + b'\n')
    sys.stdout.flush()

def getsaying():
    global SAY_P, SAY_Q
    while 1:
        say = getsayingbytext(mode='')
        SAY_Q.put(say)

def getsayingbytext(text='', mode='r'):
    global SAY_P
    with SAY_LCK:
        text = (mode + ' '.join(mosesproxy.cut(zhconv.convert(text, 'zh-hans'), HMM=False)[:60]).strip()).encode('utf-8') + b'\n'
        try:
            SAY_P.stdin.write(text)
            SAY_P.stdin.flush()
            say = SAY_P.stdout.readline().strip().decode('utf-8')
        except BrokenPipeError:
            SAY_P = subprocess.Popen(SAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='vendor')
            SAY_P.stdin.write(text)
            SAY_P.stdin.flush()
            say = SAY_P.stdout.readline().strip().decode('utf-8')
    return say

def process(obj):
    ret, exc = None, None
    try:
        ret = COMMANDS[obj['cmd']](*obj['args'])
    except Exception:
        exc = traceback.format_exc()
    return {'id': obj['id'], 'ret': ret, 'exc': exc}

def cmd_calc(expr):
    '''/calc <expr> Calculate <expr>.'''
    # Too many bugs
    r = fx233es.Evaluate(expr)
    res = None
    if r is not None or fx233es.rformat:
        res = fx233es.PrintResult()
        if len(res) > 200:
            res = res[:200] + '...'
    return res or 'Nothing'

def cmd_py(expr):
    proc = subprocess.Popen(EVIL_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='vendor')
    try:
        result, errs = proc.communicate(expr.strip().encode('utf-8'), timeout=5)
    except Exception: # TimeoutExpired
        proc.kill()
        result, errs = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.terminate()
    result = result.strip().decode('utf-8', errors='replace')
    return result or 'None or error occurred.'

def cmd_bf(expr, datain=''):
    fd, fpath = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as temp_bf:
        temp_bf.write(''.join(c for c in expr if c in '-[>.<]+,').encode('latin_1'))
    proc = subprocess.Popen(BF_CMD + (fpath,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    datain = datain.encode('utf-8')
    try:
        result, errs = proc.communicate(datain, timeout=0.1)
    except Exception: # TimeoutExpired
        proc.kill()
        result, errs = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.terminate()
        os.remove(fpath)
    if len(result) > 1000:
        result = result[:1000] + b'...'
    result = result.decode('latin_1').encode('unicode_escape').decode('latin_1').replace('\\t', '\t').replace('\\n', '\n')
    if len(result) > 1000:
        result = result[:1000] + '...'
    return result or 'None or error occurred.'

def cmd_lisp(expr):
    proc = subprocess.Popen(LISP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='vendor')
    try:
        result, errs = proc.communicate(expr.strip().encode('utf-8'), timeout=5)
    except Exception: # TimeoutExpired
        proc.kill()
        result, errs = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.terminate()
    result = result.strip().decode('utf-8', errors='replace')
    return result or 'None or error occurred.'

def cmd_name(expr):
    surnames, names = namemodel.processinput(expr, 10)
    res = []
    if surnames:
        res.append('姓：' + ', '.join(surnames[:10]))
    if names:
        res.append('名：' + ', '.join(names[:10]))
    return '\n'.join(res)

def cmd_ime(expr):
    return zhconv.convert(simpleime.pinyininput(expr.lower()), 'zh-hans')

def cmd_fig(expr):
    r = fcgen.render(expr)
    rl = r.splitlines()
    if not r:
        return 'Missing glyph(s).'
    elif len(rl[0]) < 12 and len(rl) < 15:
        return r
    else:
        return 'Figure too big.'

def cmd_cc(expr):
    if zhconv.issimp(expr):
        return zhconv.convert(expr, 'zh-hant')
    else:
        return zhconv.convert(expr, 'zh-hans')

def cmd_cut(tinput, lang):
    if lang == 'c':
        return ' '.join(mosesproxy.jiebazhc.cut(tinput, HMM=False))
    else:
        return ' '.join(mosesproxy.cut(tinput, HMM=False))

def cmd_wyw(tinput, lang):
    if lang is None:
        cscore, mscore = zhutil.calctxtstat(tinput)
        if cscore == mscore:
            lang = None
        elif zhutil.checktxttype(cscore, mscore) == 'c':
            lang = 'c2m'
        else:
            lang = 'm2c'
    if lang:
        return mosesproxy.translate(tinput, lang, 0, 0, 0)
    else:
        return tinput

def cmd_say():
    return SAY_Q.get() or 'ERROR_BRAIN_NOT_CONNECTED'

def cmd_reply(expr):
    return getsayingbytext(expr, 'r') or 'ERROR_BRAIN_NOT_CONNECTED'

def cmd_cont(expr):
    return getsayingbytext(expr, 'c') or 'ERROR_BRAIN_NOT_CONNECTED'

COMMANDS = collections.OrderedDict((
('calc', cmd_calc),
('py', cmd_py),
('bf', cmd_bf),
('lisp', cmd_lisp),
('name', cmd_name),
('ime', cmd_ime),
('fig', cmd_fig),
('cc', cmd_cc),
('wyw', cmd_wyw),
('cut', cmd_cut),
('say', cmd_say),
('reply', cmd_reply),
('cont', cmd_cont)
))

MSG_Q = queue.Queue()
SAY_Q = queue.Queue(maxsize=50)
SAY_LCK = threading.Lock()

SAY_CMD = ('python3', 'say.py', 'chat.binlm', 'chatdict.txt', 'context.pkl')
SAY_P = subprocess.Popen(SAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='vendor')

EVIL_CMD = ('python', 'seccomp.py')
BF_CMD = ('vendor/brainfuck',)
LISP_CMD = ('python', 'lispy.py')

cmdthr = threading.Thread(target=docommands)
cmdthr.daemon = True
cmdthr.start()

saythr = threading.Thread(target=getsaying)
saythr.daemon = True
saythr.start()

# fx233es = fparser.Parser(numtype='decimal')
namemodel = chinesename.NameModel('vendor/namemodel.m')
simpleime.loaddict('vendor/pyindex.dawg', 'vendor/essay.dawg')
fcgen = figchar.BlockGenerator('vendor/wqy.pkl', '🌝🌚')

try:
    for ln in sys.stdin.buffer:
        upd = json.loads(ln.decode('utf-8'))
        MSG_Q.put(upd)
finally:
    SAY_P.terminate()
