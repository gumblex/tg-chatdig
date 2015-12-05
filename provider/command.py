#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import *
import math
import random

srandom = random.SystemRandom()
facescore = lambda x,y: 1/2*math.erfc((0.5*y-x)/(2**0.5*(0.5*y**0.5)))*100
fstable = [facescore(i, 100) for i in range(101)]
revface = lambda x: min((abs(x-v), k) for k,v in enumerate(fstable))[1]

def cmd(name, protocal=None, **kwargs):
    def wrap_function(func):
        def _decorator(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        self.commands[name] = func
        self.cmdinfo[name] = kwargs
        if not protocal:
            self.cmdinfo[name]['protocals'] = ()
        elif isinstance(protocal, str):
            self.cmdinfo[name]['protocals'] = (protocal,)
        else:
            self.cmdinfo[name]['protocals'] = tuple(protocal)
        return func
    return wrap_function

class CommandHandler:
    def __init__(self, host):
        self.host = host
        self.prefix = "/'"
        self.cmds = {k[4:]: getattr(self, k) for k in dir(self) if k.startswith('cmd_')}

    def cmdinfo(self, cmd):
        return vars(self.cmds[cmd])

    def check_protocal(self, cmd, protocal):
        ret = self.cmdinfo(cmd).get('protocal')
        if ret:
            return protocal in ret
        else:
            return True

    def onmsg(self, msg):
        text = msg.get('text')
        if not text:
            return NotImplemented
        expr = text.split(' ', 1)
        cmd = expr[0]
        if msg['text'][0] in self.prefix:
            fn = self.getcmd(cmd[1:])
            if fn and msg.protocal in self.cmdinfo(cmd[1:]).get('protocal', ()):
                return fn(msg, expr[1])
            else:
                return NotImplemented
        else:
            fn = self.cmds.get('')
            if fn and msg.protocal in self.cmdinfo(cmd[1:]).get('protocal'):
                return fn(msg, text)
            else:
                return NotImplemented

    def cmd_(self):
        return NotImplemented

    def cmd_hello(self):
        return 'Hello!'

    def cmd_233(self, msg, expr, query=None):
        if query:
            num = query['num']
        else:
            try:
                num = max(min(int(expr), 100), 1)
            except Exception:
                num = 1
        w = math.ceil(num ** .5)
        h, rem = divmod(num, w)
        txt = '\n'.join(''.join(srandom.choice('🌝🌚') for i in range(w)) for j in range(h))
        if rem:
            txt += '\n' + ''.join(srandom.choice('🌝🌚') for i in range(rem))
        wcount = txt.count('🌝')
        if num > 9:
            txt += '\n' + '(🌝%d/🌚%d' % (wcount, num - wcount)
            if num > 41:
                txt += ', 🌝%.2f%%' % facescore(wcount, num)
            txt += ')'
        return txt

    def cmd_help(self, msg, expr):
        if msg['protocal'] == 'tgbot':
            if expr:
                if expr in self.cmds:
                    h = self.cmds[expr].__doc__
                    if h:
                        return h
                    else:
                        return 'Help is not available for ' + expr
                else:
                    return 'Command not found.'
            elif chatid == -self.host.tgbot.cfg['groupid']:
                return 'Full help disabled in this group.'
            elif chatid > 0:
                return '\n'.join(uniq(cmd.__doc__ for cmdname, cmd in self.cmds.items() if cmd.__doc__ and self.check_protocal(cmdname, 'tgbot')))
            else:
                return '\n'.join(uniq(cmd.__doc__ for cmdname, cmd in self.cmds.items() if cmd.__doc__ and self.check_protocal(cmdname, 'tgbot') and not self.cmdinfo(cmdname).get('tgpriv')))
        else:
            return NotImplemented

