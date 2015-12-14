#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import re
import math

sys.stderr = sys.stdout
with sys.stdin as r:
    prog = r.read()

try:
    ret = eval(prog)
    if ret is not None:
        print(ret)
except SyntaxError:
    exec(prog)
