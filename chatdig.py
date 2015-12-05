#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import signal
import logging
import operator
import threading
import functools
import collections
import concurrent.futures

import provider

__version__ = '2.0'

class MsgServer():
    def __init__(self, protocals, loggers):
        self.protocals = protocals
        self.loggers = loggers
        self.threads = []
        self.cmdh = provider.CommandHandler(self)
        self.executor = concurrent.futures.ThreadPoolExecutor(10)

    def newmsg(self, msg):
        fs = [self.executor.submit(h.onmsg, msg) for h in self.loggers]
        fs.append(self.executor.submit(self.cmdh.onmsg, msg))
        for f in fs:
            try:
                f.result(timeout=10)
            except Exception:
                logging.exception('Error processing msg: %s' % msg)

    def logmsg(self, msg):
        logging.info(msg)

    def signal(self, signum, frame):
        if signum in (signal.SIGINT, signal.SIGTERM):
            logging.info('Got signal %s: exiting...' % signum)
            self.teardown()
        elif signum == signal.SIGUSR1:
            logging.info('Got signal %s: committing db...' % signum)

    def setup(self):
        self.p = {}
        for p in self.loggers:
            self.p[p.name] = p
            p.setup(self)
        for p in self.protocals:
            self.p[p.name] = p
            p.setup(self)
            thr = threading.Thread(target=p.run)
            thr.daemon = True
            thr.start()
            self.threads.append(thr)

    def teardown(self):
        [p.teardown() for p in self.protocals]
        [p.teardown() for p in self.loggers]
        self.executor.shutdown()

    def __getattr__(self, name):
        try:
            return self.p[name]
        except KeyError:
            raise AttributeError


protocals = [
    #provider.TCPSocketProtocal('0.0.0.0', 12342),
    #provider.TelegramBotProtocal(),
    provider.DummyProtocal()
]
loggers = [
    provider.SimpleLogger()
]


loglevel = logging.DEBUG if sys.argv[-1] == '-d' else logging.INFO
logging.basicConfig(stream=sys.stdout, format='# %(asctime)s [%(levelname)s] %(message)s', level=loglevel)

MSrv = MsgServer(protocals, loggers)
MSrv.setup()
signal.signal(signal.SIGINT, MSrv.signal)
signal.signal(signal.SIGTERM, MSrv.signal)
signal.signal(signal.SIGUSR1, MSrv.signal)
logging.info("Satellite launched. Pid: %s" % os.getpid())
try:
    for thr in MSrv.threads:
        thr.join()
finally:
    MSrv.teardown()
