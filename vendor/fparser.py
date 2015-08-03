#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A calculator module to simulate the functions of some natural-display scientific calculators."""

# TODO: move descriptions to doc string.
# type, priority, parameters
# type:0:Const,1:Function,2:Operator-ltr,3:Operator-rtl
# 4:Delimiter,5:BracketEnd,6:End
# Fraction type (Complex):
#    __    __      __    __
# a1√c1+d1√e1   a2√c2+d2√e2
# ----------- + -----------i
#      b1            b2

# class parser, errors, frac, cfrac, comp, cmplx, stat, base, eqn, table,
# matrix, vector, ineq, ratio

# Err type: MathERROR, SyntaxERROR, DimensionERROR, CannotSolve,
# VariableERROR, TimeOut, ArgumentERROR

__verfunc__ = 0.95
__verlib__ = 0.78
__version__ = int(__verfunc__ * 100 + __verlib__ * 10)

# TODO: ->X. STAT, TABLE, general interactive IO

import math
import cmath
import random
from decimal import *
try:
    import readline
except ImportError:
    pass

D = Decimal

decctx_casiofxes = Context(
    prec=15,
    rounding=ROUND_HALF_UP,
    Emin=-99,
    Emax=99,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[
        DivisionByZero,
        Underflow,
        Clamped,
        InvalidOperation,
        Overflow])

decctx_canonfsga = Context(
    prec=18,
    rounding=ROUND_HALF_UP,
    Emin=-99,
    Emax=99,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[
        DivisionByZero,
        Underflow,
        Clamped,
        InvalidOperation,
        Overflow])

decctx_general = Context(
    prec=28,
    rounding=ROUND_HALF_UP,
    Emin=-100000000,
    Emax=100000000,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[
        DivisionByZero,
        Underflow,
        Clamped,
        InvalidOperation,
        Overflow])

setcontext(decctx_canonfsga)

# type, priority, parameters
# follow f-789sga standard.
op = {"Rand": [0, 1, 0],
      "i": [0, 1, 0],
      "pi": [0, 1, 0],
      "e": [0, 1, 0],
      "_": [0, 1, 0],
      "(": [1, 2, 1],
      "Pol(": [1, 3, 2],
      "Rec(": [1, 3, 2],
      "integr(": [1, 3, 3],
      "d/dx(": [1, 3, 2],
      "Sigma(": [1, 3, 3],
      "Pi(": [1, 3, 3],
      "solve(": [1, 3, (1, 2, 3, 4)],
      "nsolve(": [1, 3, (2, 3)],
      "P(": [1, 3, 1],
      "Q(": [1, 3, 1],
      "R(": [1, 3, 1],
      "Mod(": [1, 3, 2],
      "sin(": [1, 3, 1],
      "cos(": [1, 3, 1],
      "tan(": [1, 3, 1],
      "cot(": [1, 3, 1],
      "sec(": [1, 3, 1],
      "csc(": [1, 3, 1],
      "asin(": [1, 3, 1],
      "acos(": [1, 3, 1],
      "atan(": [1, 3, 1],
      "sinh(": [1, 3, 1],
      "cosh(": [1, 3, 1],
      "tanh(": [1, 3, 1],
      "asinh(": [1, 3, 1],
      "acosh(": [1, 3, 1],
      "atanh(": [1, 3, 1],
      "log(": [1, 3, (1, 2)],
      "ln(": [1, 3, 1],
      "exp(": [1, 3, 1],
      "sqrt(": [1, 3, 1],
      "cbrt(": [1, 3, 1],
      "Arg(": [1, 3, 1],
      "Abs(": [1, 3, 1],
      "Conjg(": [1, 3, 1],
      "Real(": [1, 3, 1],
      "Imag(": [1, 3, 1],
      "Not(": [1, 3, 1],
      "Neg(": [1, 3, 1],
      "Det(": [1, 3, 1],
      "Trn(": [1, 3, 1],
      "Ide(": [1, 3, 1],
      "Adj(": [1, 3, 1],
      "Inv(": [1, 3, 1],
      "Round(": [1, 3, 1],
      "int(": [1, 3, 1],
      "factor(": [1, 3, 1],
      "LCM(": [1, 3, tuple(range(2, 11))],
      "GCD(": [1, 3, tuple(range(2, 11))],
      "Q...r(": [1, 3, 2],
      "i~Rand(": [1, 3, 2],
      "table(": [1, 3, (2, 3, 4)],
      "^": [3, 4, 2],
      "!": [2, 4, 1],
      "`": [2, 4, 2],
      "(deg)": [2, 4, 1],
      "(rad)": [2, 4, 1],
      "(gra)": [2, 4, 1],
      "xroot": [2, 4, 2],
      ">t": [2, 4, 1],
      "%": [2, 4, 1],
      "\\": [2, 5, 2],
      "~": [3, 6, 1],
      "d": [3, 6, 1],
      "h": [3, 6, 1],
      "b": [3, 6, 1],
      "o": [3, 6, 1],
      "~x": [2, 7, 1],
      "~y": [2, 7, 1],
      "~x1": [2, 7, 1],
      "~x2": [2, 7, 1],
      "&": [2, 8, 2],
      "nPr": [2, 9, 2],
      "nCr": [2, 9, 2],
      "<": [2, 9, 2],
      "@": [2, 10, 2],
      "*": [2, 11, 2],
      "/": [2, 11, 2],
      "+": [2, 12, 2],
      "-": [2, 12, 2],
      "and": [2, 13, 2],
      "or": [2, 14, 2],
      "xor": [2, 14, 2],
      "xnor": [2, 14, 2],
      "=": [3, 15, 2],
      ")": [5, 15, 1],
      "M+": [6, 15, 1],
      "M-": [6, 15, 1],
      "->": [6, 15, 2],
      ">rtheta": [6, 15, 1],
      ">a+bi": [6, 15, 1],
      ":": [6, 15, 2],
      ",": [4, 2, 2]}
opalias_raw = {"Rand": ["Ran#", "rand", "ran#"],
               "_": ["ans"],
               "Pol(": ["pol("],
               "Rec(": ["rec("],
               "integr(": ["Integr(", "Integral(", "Integrate(", "integral(", "integrate(", "∫("],
               "d/dx(": ["diff("],
               "Sigma(": ["sigma(", "Sum(", "sum(", "∑("],
               # "pi(" == ["pi","&","("]
               "Pi(": ["Product(", "product(", "∏("],
               "nsolve(": ["newton("],
               "P(": ["p("],
               "Q(": ["q("],
               "R(": ["r("],
               "Mod(": ["mod("],
               "asin(": ["arcsin(", "sin-1("],
               "acos(": ["arccos(", "cos-1("],
               "atan(": ["arctan(", "tan-1("],
               "asinh(": ["arcsinh(", "sinh-1("],
               "acosh(": ["arccosh(", "cosh-1("],
               "atanh(": ["arctanh(", "tanh-1("],
               "log(": ["log10("],
               "sqrt(": ["sq(", "√("],
               "Arg(": ["arg("],
               "Abs(": ["abs("],
               "Conjg(": ["conjg(", "Conjugate(", "conjugate("],
               "Real(": ["real(", "Re(", "re("],
               "Imag(": ["imag(", "Im(", "im("],
               "Not(": ["not("],
               "Neg(": ["neg("],
               "Det(": ["det("],
               "Trn(": ["trn("],
               "Ide(": ["ide("],
               "Adj(": ["adj("],
               "Inv(": ["inv("],
               "Round(": ["rnd(", "Rnd(", "round(", "float("],
               "factor(": ["fact(", "pfactor(", "pfact("],
               "LCM(": ["lcm("],
               "GCD(": ["gcd("],
               "Q...r(": ["q...r(", "Q…r(", "q…r(", "qr("],
               "i~Rand(": ["i~rand(", "irand(", "RanInt#(", "ranint#(", "ranint("],
               "(deg)": ["(d)"],
               "(rad)": ["(r)"],
               "(gra)": ["(g)"],
               "xroot": ["root", "x√"],
               "h": ["0x"],
               "nPr": ["npr"],
               "nCr": ["ncr"],
               "<": ["∠"],
               "^": ["**"],
               "and": ["AND", "&&"],
               "or": ["OR", "||"],
               "xor": ["XOR"],
               "xnor": ["XNOR"],
               "M+": ["m+"],
               "M-": ["m-"],
               "->": ["→"],
               ">rtheta": [">r∠θ"],
               ">a+bi": [">abi"]}
opalias = {}
for i in op:
    opalias[i] = i
for i in opalias_raw:
    opalias[i] = i
    for j in opalias_raw[i]:
        opalias[j] = i
del opalias_raw, i, j


def gcd(*numbers):
    """Calculate the Greatest Common Divisor of the numbers."""
    if len(numbers) == 2:
        a, b = numbers
        while b:
            a, b = b, a % b
        return a
    elif len(numbers) < 2:
        raise TypeError(
            'gcd expected at least 2 arguments, got ' + str(len(numbers)))
    else:
        val = numbers[0]
        for i in numbers[1:]:
            while i:
                val, i = i, val % i
        return val


def lcm(*numbers):
    """Calculate the Lowest Common Multiple of the numbers."""
    if len(numbers) == 2:
        return numbers[0] * numbers[1] // gcd(numbers[0], numbers[1])
    elif len(numbers) < 2:
        raise TypeError(
            'lcm expected at least 2 arguments, got ' + str(len(numbers)))
    else:
        val = numbers[0]
        for i in numbers[1:]:
            val = val * i // gcd(val, i)
        return val


def pfact(n):
    """Returns all the prime factors of a positive integer."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def frange(start, stop, step):
    while start < stop:
        yield start
        start += step


def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value


def decifmt(n, mode='norm', dg=1):
    # TODO: /####
    if mode == 'fix':
        return str(round(n, dg))
    elif mode == 'sci':
        ctx = getcontext().copy()
        ctx.prec = dg
        return ctx.to_sci_string(n.normalize(ctx))
    else:
        if dg == 2:
            pass
        else:  # ==1
            pass


def pi():
    """Compute Pi to the current precision.

    >>> print(pi())
    3.141592653589793238462643383

    """
    getcontext().prec += 2  # extra digits for intermediate steps
    three = Decimal(3)      # substitute "three=3.0" for regular floats
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s               # unary plus applies the new precision


def exp(x):
    """Return e raised to the power of x.  Result type matches input type.

    >>> print(exp(Decimal(1)))
    2.718281828459045235360287471
    >>> print(exp(Decimal(2)))
    7.389056098930650227230427461
    >>> print(exp(2.0))
    7.38905609893
    >>> print(exp(2+0j))
    (7.38905609893+0j)

    """
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 1
        fact *= i
        num *= x
        s += num / fact
    getcontext().prec -= 2
    return +s

# constant instead of function
getcontext().prec += 2
c_pi = pi()
c_e = exp(D(1))
getcontext().prec -= 2


def cos(x):
    """Return the cosine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(cos(Decimal('0.5')))
    0.8775825618903727161162815826
    >>> print(cos(0.5))
    0.87758256189
    >>> print(cos(0.5+0j))
    (0.87758256189+0j)

    """
    getcontext().prec += 2
    if abs(x) > 2 * c_pi:
        x = x % (2 * c_pi)
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


def sin(x):
    """Return the sine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(sin(Decimal('0.5')))
    0.4794255386042030002732879352
    >>> print(sin(0.5))
    0.479425538604
    >>> print(sin(0.5+0j))
    (0.479425538604+0j)

    """
    getcontext().prec += 2
    if abs(x) > 2 * c_pi:
        x = x % (2 * c_pi)
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


def cosh(x):
    getcontext().prec += 2
    i, lasts, s, fact, num = 0, 0, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        s += num / fact
    getcontext().prec -= 2
    return +s


def sinh(x):
    i, lasts, s, fact, num = 1, 0, x, 1, x
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        s += num / fact
    getcontext().prec -= 2
    return +s

# tan = lambda x : sin(x) / cos(x)
# cot = lambda x : cos(x) / sin(x)
# sec = lambda x : 1 / cos(x)
# csc = lambda x : 1 / sin(x)


def det(l):
    n = len(l)
    if (n > 2):
        i, t, sum = 1, 0, 0
        while t <= n - 1:
            d = {}
            t1 = 1
            while t1 <= n - 1:
                m = 0
                d[t1] = []
                while m <= n - 1:
                    if (m == t):
                        u = 0
                    else:
                        d[t1].append(l[t1][m])
                    m += 1
                t1 += 1
            l1 = [d[x] for x in d]
            sum = sum + i * (l[0][t]) * (det(l1))
            i = i * (-1)
            t += 1
        return sum
    elif n == 2:
        return (l[0][0] * l[1][1] - l[0][1] * l[1][0])
    else:
        raise TypeError('det expected len(l)>=2, got ' + str(len(l)))


class Calculator:

    """The main Calculator class."""

    def __init__(self, mode='comp', vars={}, numtype='frac'):
        self.vars = {
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
            'X': 0, 'Y': 0, 'M': 0, 'Ans': 0, '_0': 0, '_1': 0,
            '_2': 0, '_3': 0, '_4': 0, '_5': 0, '_6': 0, '_7': 0,
            '_8': 0, '_9': 0, '_10': 0, 'MatA': [], 'MatB': [],
            'MatC': [], 'MatD': [], 'MatAns': [], 'VctA': [],
            'VctB': [], 'VctC': [], 'VctD': [], 'VctAns': []}
        self.mode = mode
        self.result = None
        self.format = None
        self.numtype = numtype
        # frac, cfrac, float, int, decimal
        if vars:
            self.vars.update(vars)

    def ntype(self, num, numtype=None):
        "Converts a number to the proper number type according to self.numtype"
        if not numtype:
            numtype = self.numtype
        if isinstance(num, frac):
            if numtype == 'frac':
                return num
            else:
                n = num.todec()
        elif isinstance(num, cfrac):
            if numtype == 'cfrac':
                return num
            else:
                n = num.real.todec()
        elif isinstance(num, str):
            n = D(num)
        elif isinstance(num, Decimal):
            if numtype == 'decimal':
                return +num
            else:
                n = num
        else:
            n = D(num)
        if numtype == 'frac':
            return frac(num)
        elif numtype == 'cfrac':
            return cfrac(num)
        elif numtype == 'float':
            return float(num)
        elif numtype == 'int':
            return int(num)
        elif numtype == 'decimal':
            return +n
        else:
            return +n


class MathERROR(Exception, Calculator):

    '''The Math ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'MathERROR(%s)' % self.loc


class SyntaxERROR(Exception, Calculator):

    '''The Syntax ERROR type.'''
    # TODO: err info

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'SyntaxERROR(%s)' % self.loc


class KbdBreak(Exception, Calculator):

    '''The Keyboard Break ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'KbdBreak(%s)' % self.loc


class DimensionERROR(Exception, Calculator):

    '''The Dimension ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'DimensionERROR(%s)' % self.loc


class CannotSolve(Exception, Calculator):

    """The Can't Solve ERROR type."""

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'CannotSolve(%s)' % self.loc


class VariableERROR(Exception, Calculator):

    '''The Variable ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'VariableERROR(%s)' % self.loc


class TimeOut(Exception, Calculator):

    '''The Time Out ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'TimeOut(%s)' % self.loc


class ArgumentERROR(Exception, Calculator):

    '''The Argument ERROR type.'''

    def __init__(self, loc=0):
        Exception.__init__(self)
        self.loc = loc

    def __repr__(self):
        return 'ArgumentERROR(%s)' % self.loc


class frac(Calculator):

    """The basic number type the calculator uses.
      _   _
    a√c+d√e
    -------
       b
    t (type):	1:int, 2:decimal, 4:frac
    """

    def __init__(self, *argn):
        self.a, self.b, self.c, self.d, self.e = 0, 1, 1, 0, 1
        self.t = 1
        if len(argn) == 1:
            num = argn[0]
            if not num:
                return
            elif isinstance(num, frac):
                self.a, self.b, self.c, self.d, self.e = num.a, num.b, num.c, num.d, num.e
            elif isinstance(num, cfrac):
                self.a, self.b, self.c, self.d, self.e = num.real.a, num.real.b, num.real.c, num.real.d, num.real.e
            elif isinstance(num, int):
                self.a = num
                return
            elif isinstance(num, float):
                if num.is_integer():
                    self.a = int(num)
                    return
                else:
                    self.a, self.b = self.limit_denominator(
                        num.as_integer_ratio())
                    if abs(self.b) > 10000:
                        self.a, self.b = D(num), 1
            elif isinstance(num, Decimal):
                if int(num) == num:
                    self.a = int(num)
                    return
                else:
                    self.a, self.b = self.limit_denominator(
                        (int(num.scaleb(-num.as_tuple().exponent)), 10**(-num.as_tuple().exponent)))
                    if abs(self.b) > 10000:
                        self.a, self.b = num, 1
            elif isinstance(num, complex):
                if num.real.is_integer():
                    self.a = int(num.real)
                    return
                else:
                    self.a, self.b = self.limit_denominator(
                        num.real.as_integer_ratio())
                    if abs(self.b) > 10000:
                        self.a, self.b = D(num.real), 1
            elif isinstance(num, str):
                try:
                    n = D(num).normalize()
                except:
                    raise TypeError('invalid str for number')
                if int(n) == n:
                    self.a = int(n)
                    return
                else:
                    self.a, self.b = self.limit_denominator(
                        (int(n.scaleb(-n.as_tuple().exponent)), 10**(-n.as_tuple().exponent)))
                    if abs(self.b) > 10000:
                        self.a, self.b = n, 1
        elif len(argn) == 2:
            self.a, self.b = int(argn[0]), int(argn[1])
        elif len(argn) == 3:
            self.a, self.b, self.c = int(argn[0]), int(argn[1]), int(argn[2])
        elif len(argn) == 4:
            self.a, self.b, self.c, self.d = int(
                argn[0]), int(
                argn[1]), int(
                argn[2]), int(
                argn[3])
        elif len(argn) >= 5:
            self.a, self.b, self.c, self.d, self.e = int(
                argn[0]), int(
                argn[1]), int(
                argn[2]), int(
                argn[3]), int(
                    argn[4])
        self.normalize()

    def todec(self):
        if self.t == 4:
            return (D(self.a) * D(self.c).sqrt() + D(self.d)
                    * D(self.e).sqrt()) / D(self.b)
        else:
            return D(self.a)

    def __float__(self):
        return float(self.todec())

    def __int__(self):
        if self.t == 1:
            return int(self.a)
        else:
            return int(self.todec())

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, n=0):
        return round(self.__float__(), n)

    def __str__(self):
        return str(self.todec())

    def pretty(self, mathdisp=True, before='', after=''):
        self.normalize()
        if self.t == 2:
            # TODO: Different in NORM, FIX, SCI
            return before + str(self.todec().normalize()) + after
        elif self.t == 1:
            return before + str(self.todec()) + after
        elif mathdisp:
            na, nb, nc, nd, ne = str(int(self.a)), str(int(self.b)), str(
                int(self.c)), str(int(self.d)), str(int(self.e))
            l0, l1, lf, l2 = '', '', '', ''
            wbefore, wafter = ' ' * len(before), ' ' * len(after)
            if self.d == 0:
                if self.c == 1:
                    fracline = max(len(na), len(nb))
                    if self.a > 0:

                        return before + ' ' * \
                            ((fracline - len(na)) // 2) + na + '\n' + '-' * fracline + \
                            '\n' + ' ' * \
                            ((fracline - len(nb)) // 2) + nb + after
                    else:
                        return before + ' ' * ((fracline - len(str(int(-self.a)))) // 2 + 2) + str(
                            int(-self.a)) + '\n- ' + '-' * fracline + '\n' + ' ' * ((fracline - len(nb)) // 2 + 2) + nb + after
                else:
                    if abs(self.a) != 1:
                        l0 = ' ' * len(str(abs(int(self.a))))
                        l1 = str(abs(int(self.a)))
                    l1 += '√' + nc
                    l0 += ' ' + '_' * (len(nc))
                    if self.b != 1:
                        fracline = max(len(l1), len(nb))
                        l0 = ' ' * ((fracline - len(l1)) // 2) + l0
                        l1 = ' ' * ((fracline - len(l1)) // 2) + l1
                        l2 = ' ' * ((fracline - len(nb)) // 2) + nb
                        lf = '-' * fracline
                        if self.a < 0:
                            l0 = '  ' + l0
                            l1 = '  ' + l1
                            lf = '- ' + lf
                            l2 = '  ' + l2
                        return '%s\n%s\n%s\n%s' % (
                            wbefore + l0 + wafter, wbefore + l1 + wafter, before + lf + after, wbefore + l2 + wafter)
                    else:
                        if self.a < 0:
                            l0 = ' ' + l0
                            l1 = '-' + l1
                        return '%s\n%s' % (
                            wbefore + l0 + wafter, before + l1 + after)
            else:
                if self.a != 1:
                    l0 = ' ' * len(na)
                    l1 = na
                if self.c != 1:
                    l0 += ' ' + '_' * (len(nc))
                    l1 += '√' + nc
                if self.d != 0:
                    if self.e == 1:
                        if self.a > 0:
                            l0 = ' ' + l0
                            l1 = '+' + l1
                        l0 = ' ' * len(nd) + l0
                        l1 = nd + l1
                    else:
                        if self.d != 0:
                            if self.d == 1:
                                nd = '+'
                            elif self.d == -1:
                                nd = '-'
                            elif self.d > 0:
                                nd = '+' + nd
                            l0 += ' ' * len(nd)
                            l1 += nd
                            l0 += ' ' + '_' * (len(ne))
                            l1 += '√' + ne
                if self.b != 1:
                    fracline = max(len(l1), len(nb))
                    lf = '-' * fracline
                    l0 = ' ' * ((fracline - len(l1)) // 2) + l0
                    l1 = ' ' * ((fracline - len(l1)) // 2) + l1
                    l2 = ' ' * ((fracline - len(nb)) // 2) + nb
                    return '%s\n%s\n%s\n%s' % (
                        wbefore + l0 + wafter, wbefore + l1 + wafter, before + lf + after, wbefore + l2 + wafter)
                else:
                    return '%s\n%s' % (
                        wbefore + l0 + wafter, before + l1 + after)
        else:
            self.normalize()
            if self.c == self.e == 1:
                rstr = str(int(self.a))
                if self.b != 1:
                    rstr += "/" + str(self.b)
            else:
                if self.e == 1:
                    if self.d == 0:
                        rstr = "%s*sqrt(%s)" % (int(self.a), self.c)
                    else:
                        rstr = "%s+%s*sqrt(%s)" % (self.d, int(self.a), self.c)
                else:
                    rstr = "%s*sqrt(%s)+%s*sqrt(%s)" % (int(self.a),
                                                        self.c,
                                                        self.d,
                                                        self.e)
                if self.b != 1:
                    rstr = "(%s)/%s" % (rstr, str(self.b))
            # improve
            return before + rstr + after

    def __repr__(self):
        return "frac(%s,%s,%s,%s,%s)" % (
            self.a, self.b, self.c, self.d, self.e)

    def __lt__(self, other):
        return self.__float__() < float(other)

    def __le__(self, other):
        return self.__float__() <= float(other)

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self.todec() == frac(other).todec()
        except:
            return False

    def __ne__(self, other):
        if other is None:
            return True
        try:
            return self.todec() != frac(other).todec()
        except:
            return True

    def __gt__(self, other):
        return self.__float__() > float(other)

    def __ge__(self, other):
        return self.__float__() >= float(other)

    def __hash__(self):
        return hash(self.todec())

    def __bool__(self):
        return self.a != 0 and self.c != 0 or self.d != 0 and self.e != 0

    def __add__(self, other):
        self.normalize()
        y = frac(other)
        if (self.t + y.t) in (2, 3, 4, 6):
            return frac(self.todec() + y.todec())
        co = dict((i, 0) for i in (self.c, self.e, y.c, y.e))
        div = self.b * y.b
        co[self.c] += self.a * y.b
        co[self.e] += self.d * y.b
        co[y.c] += y.a * self.b
        co[y.e] += y.d * self.b
        for i in co.copy():
            if co[i] == 0:
                del co[i]
        if not co:
            return frac(0)
        elif len(co) > 2:
            return frac(self.todec() + y.todec())
        elif len(co) == 2:
            n = tuple(co.items())
            return frac(n[0][1], div, n[0][0], n[1][1], n[1][0])
        else:
            n = tuple(co.items())
            return frac(n[0][1], div, n[0][0], 0, 1)

    def __sub__(self, other):
        return self.__add__(frac(other).__neg__())

    def __mul__(self, other):
        self.normalize()
        y = frac(other)
        if (self.t + y.t) in (2, 3, 4, 6):
            return frac(self.todec() * y.todec())
        if self.c == self.e == 1:
            return frac(y.a * self.a, y.b * self.b, y.c, y.d * self.a, y.e)
        elif y.c == y.e == 1:
            return frac(
                self.a * y.a, self.b * y.b, self.c, self.d * y.a, self.e)
        if self.d == 0:
            self.e = y.e
        elif y.d == 0:
            y.e = self.e
        if self.c == y.c and self.e == y.e:
            return frac(self.a * y.d + y.a * self.d,
                        self.b * y.b, self.c * self.e,
                        self.a * y.a * self.c + self.d * y.d * self.e, 1)
        else:
            return frac(self.todec() * y.todec())

    def __truediv__(self, other):
        if not other:
            raise MathERROR
        self.normalize()
        y = frac(other)
        if (self.t + y.t) in (2, 3, 4, 6):
            return frac(self.todec() / y.todec())
        if self.e == y.e == 1:
            return frac(self.a * y.b, y.a * self.b * y.c, self.c * y.c)
        elif self.c == self.e == 1:
            # implict *.d==0
            return frac(y.a * y.b * self.a, y.a**2 * y.c * self.b -
                        y.d**2 * y.e * self.b, y.c, y.d * y.b * self.a, y.e)
        elif y.c == y.e == 1:
            return frac(
                self.a * y.b, self.b * y.a, self.c, self.d * y.b, self.e)
        if self.d == 0:
            self.e = y.e
        elif y.d == 0:
            y.e = self.e
        if self.c == y.c and self.e == y.e:
            return frac(y.a * self.b * y.b * self.d - self.a * self.b * y.b * y.d,
                        y.a**2 * self.b**2 * self.c -
                        self.b**2 * y.d**2 * self.e,
                        self.c * self.e,
                        self.a * y.a * self.b * y.b * self.c - self.b * y.b * self.d * y.d * self.e, 1)
        else:
            return frac(self.todec() / y.todec())

    def __floordiv__(self, other):
        return frac(self.todec() // frac(other).todec())

    def __mod__(self, other):
        y = frac(other)
        return frac(self.todec() % y.todec())

    def __divmod__(self, other):
        y = frac(other)
        return frac(divmod(self.todec(), y.todec()))

    def __pow__(self, other, modulo=None):
        if self == other == 0:
            raise MathERROR
        if int(other) == other:
            result = frac(1)
            m = int(abs(other))
            for i in range(m):
                result *= self
            if other < 0:
                result = 1 / result
            if modulo:
                result %= modulo
            return result
        elif other == .5:
            if self < 0:
                return cfrac(self)**.5
            self.normalize()
            if self.t == 2:
                return frac(self.todec().sqrt())
            elif self.c == 1 and self.e == 1:
                if modulo:
                    return frac((self.todec().sqrt()) % modulo)
                else:
                    return frac(1, self.b, (self.a + self.d) * self.b)
        else:
            return pow(self.todec(), frac(other).todec(), modulo)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other).__neg__()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if not self:
            raise MathERROR
        y = frac(other)
        if (self.t + y.t) in (2, 3, 4, 6):
            return frac(y.todec() / self.todec())
        else:
            return y.__mul__(frac(
                self.a * self.b, self.a**2 * self.c - self.d**2 * self.e, self.c, self.d * self.b, self.e))

    def __rfloordiv__(self, other):
        return frac(frac(other).todec() // self.todec())

    def __rmod__(self, other):
        y = frac(other)
        return frac(y.todec() % self.todec())

    def __rdivmod__(self, other):
        y = frac(other)
        return frac(divmod(y.todec(), self.todec()))

    def __rpow__(self, other):
        return frac(other).__pow__(self)

    def __neg__(self):
        if self.t == 4:
            return frac(-self.a, self.b, self.c, -self.d, self.e)
        else:
            return frac(-self.a)

    def __pos__(self):
        return self

    def conjugate(self):
        return self

    def copy(self):
        return frac(self)

    def real(self):
        return self
    real = property(real)

    def imag(self):
        return frac(0)
    imag = property(imag)

    def __abs__(self):
        if self.__float__() < 0:
            return self.__neg__()
        else:
            return self

    def isfloat(self):
        return int(self.a) != self.a

    def simplify(self, num=None):
        if (self.b, self.c, self.d, self.e) == (1, 1, 0, 1):
            if int(self.a) == self.a:
                self.t = 1
            else:
                self.t = 2
            return False
        else:
            self.t = 4

        def simplifysqrt(num):
            if num == 0:
                return (0, 1)
            elif num < 0:
                raise ValueError("negative number cannot be sqrt'ed.")
            outside_root = 1
            inside_root = num
            d = 2
            while (d * d <= inside_root):
                if (inside_root % (d * d) == 0):
                    # inside_root evenly divisible by d * d
                    inside_root = inside_root // (d * d)
                    outside_root = outside_root * d
                else:
                    d = d + 1
            return (int(outside_root), int(inside_root))

        if num is None:
            if self.isfloat():
                return
            if self.a == 0:
                self.c == 1
            if self.d == 0:
                self.e == 1
            tmpsqrt = simplifysqrt(self.c)
            self.a, self.c = self.a * tmpsqrt[0], tmpsqrt[1]
            tmpsqrt = simplifysqrt(self.e)
            self.d, self.e = self.d * tmpsqrt[0], tmpsqrt[1]
            if self.c == self.e:
                self.a, self.d, self.e = self.a + self.d, 0, 1
            ngcd = gcd(self.a, self.b, self.d)
            self.a, self.b, self.d = self.a // ngcd, self.b // ngcd, self.d // ngcd
            return True
        elif isinstance(num, frac):
            if num.isfloat():
                return num
            ngcd = gcd(num.a, num.b, num.d)
            return frac(
                num.a // ngcd, num.b // ngcd, num.c, num.d // ngcd, num.e)
        elif isinstance(num, tuple):
            if len(num) == 2:
                ngcd = gcd(num[0], num[1])
                return (num[0] // ngcd, num[1] // ngcd)
            else:
                raise ValueError("num must be a frac or a (a,b) tuple")
        else:
            raise ValueError("num must be a frac or a (a,b) tuple")

    def normalize(self):
        if not self.simplify():
            return
        if not self.__bool__():
            self.a, self.b, self.c, self.d, self.e = 0, 1, 1, 0, 1
        elif int(self) == self:
            self.a, self.b, self.c, self.d, self.e = int(self), 1, 1, 0, 1
        elif int(self.b) != self.b or int(self.c) != self.c or int(self.d) != self.d or int(self.e) != self.e:
            self.a, self.b, self.c, self.d, self.e = self.todec(), 1, 1, 0, 1
        else:
            if self.b < 0:
                self.a, self.b, self.d = -self.a, -self.b, -self.d
            if self.c == self.e == 1:
                self.a, self.c, self.d, self.e = self.a + self.d, 1, 0, 1
            elif self.e > self.c:
                self.a, self.c, self.d, self.e = self.d, self.e, self.a, self.c
        if abs(self.b) > 10000:
            self.a, self.b, self.c, self.d, self.e = self.todec(), 1, 1, 0, 1

    def limit_denominator(self, fractuple, max_denominator=None):
        """Closest Fraction to self with denominator at most max_denominator."""
        # Algorithm notes: For any real number x, define a *best upper
        # approximation* to x to be a rational number p/q such that:
        #
        #   (1) p/q >= x, and
        #   (2) if p/q > r/s >= x then s > q, for any rational r/s.
        #
        # Define *best lower approximation* similarly.  Then it can be
        # proved that a rational number is a best upper or lower
        # approximation to x if, and only if, it is a convergent or
        # semiconvergent of the (unique shortest) continued fraction
        # associated to x.
        #
        # To find a best rational approximation with denominator <= M,
        # we find the best upper and lower approximations with
        # denominator <= M and take whichever of these is closer to x.
        # In the event of a tie, the bound with smaller denominator is
        # chosen.  If both denominators are equal (which can happen
        # only when max_denominator == 1 and self is midway between
        # two integers) the lower bound---i.e., the floor of self, is
        # taken.
        if not max_denominator:
            max_denominator = 10**(getcontext().prec - 2)
        if len(fractuple) != 2:
            raise ValueError(
                "fractuple should be (a,b) represents the fraction a/b")
        if max_denominator < 1:
            raise ValueError("max_denominator should be at least 1")
        ngcd = gcd(fractuple[0], fractuple[1])
        fractuple = (fractuple[0] // ngcd, fractuple[1] // ngcd)
        if fractuple[1] <= max_denominator:
            return fractuple
        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = fractuple
        while True:
            a = n // d
            q2 = q0 + a * q1
            if q2 > max_denominator:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            n, d = d, n - a * d
        k = (max_denominator - q0) // q1
        bound1 = (p0 + k * p1, q0 + k * q1)
        bound2 = (p1, q1)
        if abs(bound2[0] / bound2[1] - fractuple[0] / fractuple[1]
               ) <= abs(bound1[0] / bound1[1] - fractuple[0] / fractuple[1]):
            return bound2
        else:
            return bound1


class cfrac(Calculator):

    """Fraction type (Complex):
            __    __      __    __
     a1√c1+d1√e1   a2√c2+d2√e2
     ----------- + -----------i
              b1            b2

             real    +     imag  *i
    """

    def __init__(self, real=0, imag=None):
        if isinstance(real, complex):
            if imag:
                self.real = frac(real.real)
                self.imag = frac(imag)
            else:
                self.real = frac(real.real)
                self.imag = frac(real.imag)
        elif isinstance(real, cfrac):
            if imag:
                self.real = frac(real.real)
                self.imag = frac(imag)
            else:
                self.real = frac(real.real)
                self.imag = frac(real.imag)
        else:
            if imag:
                self.real = frac(real)
                self.imag = frac(imag)
            else:
                self.real = frac(real)
                self.imag = frac(0)

    def __float__(self):
        return float(self.real)

    def todec(self):
        return self.real.todec()

    def __int__(self):
        return int(self.real)

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def __round__(self, n=0):
        return round(D(self.real), n)

    def __str__(self):
        if not self.imag:
            return str(self.real)
        elif self.imag > 0:
            return str(self.real) + "+" + str(self.imag) + "i"
        else:
            return str(self.real) + str(self.imag) + "i"

    def pretty(self, mathdisp=True, before='', after=''):
        # TODO: use before after
        # sq(i
        if mathdisp:
            sr, si = self.real.pretty(
                True, before).split(), self.imag.pretty(True).split()
            wbefore, wafter = ' ' * len(before), ' ' * len(after)
            if self.imag:
                rstr = ["", "", "", "", ""]
                if len(sr) == 1:
                    rstr[2] = sr[0]
                    base = 2
                elif len(sr) == 2:
                    rstr[1], rstr[2] = sr[0], sr[1]
                    base = 2
                elif len(sr) == 3:
                    rstr[2], rstr[3], rstr[4] = sr[0], sr[1], sr[2]
                    base = 3
                elif len(sr) == 4:
                    rstr = sr
                    base = 3
                maxlen = len(max(rstr, key=len))
                if len(si) == 1:
                    if self.imag > 0:
                        rstr[
                            base] += " " * (maxlen - len(rstr[base]) + 1) + "+" + si[0] + "i" + after
                    else:
                        rstr[base] += " " * \
                            (maxlen - len(rstr[base]) + 1) + \
                            si[0] + "i" + after
                elif len(si) == 2:
                    rstr[base - 1] += " " * \
                        (maxlen - len(rstr[base - 1]) + 1) + si[0] + wafter
                    rstr[base] += " " * \
                        (maxlen - len(rstr[base]) + 1) + si[1] + "i" + after
                elif len(si) == 3:
                    rstr[base - 1] += " " * \
                        (maxlen - len(rstr[base - 1]) + 1) + si[0] + wafter
                    rstr[base] += " " * \
                        (maxlen - len(rstr[base]) + 1) + si[1] + "i" + after
                    rstr[base + 1] += " " * \
                        (maxlen - len(rstr[base + 1]) + 1) + si[2] + wafter
                elif len(si) == 4:
                    rstr[base - 2] += " " * \
                        (maxlen - len(rstr[base - 2]) + 1) + si[0] + wafter
                    rstr[base - 1] += " " * \
                        (maxlen - len(rstr[base - 1]) + 1) + si[1] + wafter
                    rstr[base] += " " * \
                        (maxlen - len(rstr[base]) + 1) + si[2] + "i" + after
                    rstr[base + 1] += " " * \
                        (maxlen - len(rstr[base]) + 1) + si[3] + wafter
                return "\n".join(l for l in rstr).strip()
            else:
                return self.real.pretty(True, before, after)
        else:
            if not self.imag:
                return self.real.pretty(False, before, after)
            elif self.imag > 0:
                return before + \
                    self.real.pretty(
                        False) + "+" + self.imag.pretty(False) + "i" + after
            else:
                return before + \
                    self.real.pretty(
                        False) + self.imag.pretty(False) + "i" + after

    def __repr__(self):
        return "cfrac(%s,%s)" % (repr(self.real), repr(self.imag))

    def __eq__(self, other):
        try:
            return self.__complex__() == complex(other)
        except:
            return False

    def __ne__(self, other):
        try:
            return self.__complex__() != complex(other)
        except:
            return True

    def __hash__(self):
        return hash(self.__complex__())

    def __bool__(self):
        return self.real != 0 and self.imag != 0

    def __add__(self, other):
        y = cfrac(other)
        return cfrac(self.real + y.real, self.imag + y.imag)

    def __sub__(self, other):
        y = cfrac(other)
        return cfrac(self.real - y.real, self.imag - y.imag)

    def __mul__(self, other):
        y = cfrac(other)
        return cfrac(self.real * y.real - self.imag * y.imag,
                     self.real * y.imag + self.imag * y.real)

    def __truediv__(self, other):
        y = cfrac(other)
        div = y.real**2 + y.imag**2
        return cfrac((self.real * y.real + self.imag * y.imag) /
                     div, (self.imag * y.real - self.real * y.imag) / div)

    def __pow__(self, other):
        sign = lambda x: x and (1, -1)[x < 0]
        if self == other == 0:
            raise MathERROR
        if int(other) == other:
            result = cfrac(1)
            for i in range(abs(other)):
                result *= self
            if other < 0:
                result = 1 / result
            return result
        elif other == .5:
            return cfrac(((self.__abs__() + self.real) / 2)**.5,
                         sign(self.imag) * ((self.__abs__() - self.real) / 2)**.5)
        else:
            return pow(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other).__neg__()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if not self:
            raise MathERROR
        y = cfrac(other)
        div = self.real**2 + self.imag**2
        return cfrac((y.real * self.real + y.imag * self.imag) /
                     div, (y.imag * self.real - y.real * self.imag) / div)

    def __rpow__(self, other):
        return cfrac(other).__pow__(self)

    def __neg__(self):
        return cfrac(-self.real, -self.imag)

    def __pos__(self):
        return self

    def conjugate(self):
        return cfrac(self.real, -self.imag)

    def copy(self):
        return cfrac(self)

    def __abs__(self):
        return (self.real**2 + self.imag**2)**.5

    def phase(self):
        pass

    def polar(self):
        pass

    def isreal(self):
        return self.imag == 0


class dcomplex(Calculator):

    """Complex numbers use Decimal."""

    def __init__(self, real=0, imag=None):
        if isinstance(real, complex):
            if imag:
                self.real = Decimal(real.real)
                self.imag = Decimal(imag)
            else:
                self.real = Decimal(real.real)
                self.imag = Decimal(real.imag)
        elif isinstance(real, cfrac):
            if imag:
                self.real = self._ntodec(real.real)
                self.imag = self._ntodec(imag)
            else:
                self.real = self._ntodec(real.real)
                self.imag = self._ntodec(real.imag)
        else:
            if imag:
                self.real = self._ntodec(real)
                self.imag = self._ntodec(imag)
            else:
                self.real = self._ntodec(real)
                self.imag = Decimal(0)

    def _ntodec(self, num):
        return super(Parser, self).ntype(num, 'decimal')

    def __float__(self):
        return float(self.real)

    def todec(self):
        return self.real

    def __int__(self):
        return int(self.real)

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def __round__(self, n=0):
        return round(self.real, n)

    def __str__(self):
        if not self.imag:
            return str(self.real)
        elif self.imag > 0:
            return str(self.real) + "+" + str(self.imag) + "i"
        else:
            return str(self.real) + str(self.imag) + "i"

    def pretty(self, mathdisp=False, before='', after=''):
        return before + self.__str__() + after

    def __repr__(self):
        return "dcomplex(%s,%s)" % (repr(self.real), repr(self.imag))

    def __eq__(self, other):
        try:
            return self.__complex__() == complex(other)
        except:
            return False

    def __ne__(self, other):
        try:
            return self.__complex__() != complex(other)
        except:
            return True

    def __hash__(self):
        return hash(self.__complex__())

    def __bool__(self):
        return self.real != 0 and self.imag != 0

    def __add__(self, other):
        y = dcomplex(other)
        return dcomplex(self.real + y.real, self.imag + y.imag)

    def __sub__(self, other):
        y = dcomplex(other)
        return dcomplex(self.real - y.real, self.imag - y.imag)

    def __mul__(self, other):
        y = dcomplex(other)
        return dcomplex(self.real * y.real - self.imag *
                        y.imag, self.real * y.imag + self.imag * y.real)

    def __truediv__(self, other):
        y = dcomplex(other)
        div = y.real**2 + y.imag**2
        return dcomplex((self.real * y.real + self.imag * y.imag) /
                        div, (self.imag * y.real - self.real * y.imag) / div)

    def __pow__(self, other):
        sign = lambda x: x and (1, -1)[x < 0]
        if self == other == 0:
            raise MathERROR
        if int(other) == other:
            result = dcomplex(1)
            for i in range(abs(other)):
                result *= self
            if other < 0:
                result = 1 / result
            return result
        elif other == .5:
            return dcomplex(((self.__abs__() + self.real) / 2)**.5,
                            sign(self.imag) * ((self.__abs__() - self.real) / 2)**.5)
        else:
            return pow(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other).__neg__()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if not self:
            raise MathERROR
        y = dcomplex(other)
        div = self.real**2 + self.imag**2
        return dcomplex((y.real * self.real + y.imag * self.imag) /
                        div, (y.imag * self.real - y.real * self.imag) / div)

    def __rpow__(self, other):
        return dcomplex(other).__pow__(self)

    def __neg__(self):
        return dcomplex(-self.real, -self.imag)

    def __pos__(self):
        return self

    def conjugate(self):
        return dcomplex(self.real, -self.imag)

    def copy(self):
        return dcomplex(self)

    def __abs__(self):
        return (self.real**2 + self.imag**2)**.5

    def phase(self):
        pass

    def polar(self):
        pass

    def isreal(self):
        return self.imag == 0


class Parser(Calculator):

    def __init__(self, expr=None, var={}, numtype='frac'):
        self.var = {
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
            'X': 0, 'Y': 0, 'M': 0, 'Ans': 0, '_0': 0, '_1': 0,
            '_2': 0, '_3': 0, '_4': 0, '_5': 0, '_6': 0, '_7': 0,
            '_8': 0, '_9': 0, '_10': 0}
        self.expr = None
        self.result = None
        self.rformat = None
        self.numtype = numtype
        # frac, cfrac, float, int, decimal
        if var:
            self.var.update(var)
        if expr:
            self.expr = str(expr).strip()

    def ntype(self, num, numtype=None):
        if numtype:
            return super(Parser, self).ntype(num, numtype)
        else:
            return super(Parser, self).ntype(num, self.numtype)

    def In2Post(self, lstin):
        opstack = []
        lstout = []
        for key in range(len(lstin)):
            i = lstin[key]
            itype = op.get(i[0])
            if itype:
                if itype[0] == 0:
                    lstout.append((self.ntype(self.oeval(i[0], [])), i[1]))
                elif itype[0] == 1:
                    opstack.append(i)
                elif itype[0] in (2, 3):
                    if opstack:
                        oper = op[opstack[-1][0]]
                        while oper[0] in (2, 3) and (
                                itype[0] == 2 and itype[1] >= oper[1] or itype[0] == 3 and itype[1] > oper[1]):
                            lstout.append(opstack.pop())
                            if opstack:
                                oper = op[opstack[-1][0]]
                            else:
                                break
                    opstack.append(i)
                elif itype[0] == 4:
                    try:
                        while op[opstack[-1][0]][0] != 1:
                            lstout.append(opstack.pop())
                    except IndexError:
                        raise SyntaxERROR(key)
                    if len(opstack[-1]) == 2:
                        opstack[-1] += (2,)
                    else:
                        opstack[-1] = (opstack[-1][0],
                                       opstack[-1][1],
                                       opstack[-1][2] + 1)
                elif itype[0] == 5:
                    try:
                        while op[opstack[-1][0]][0] != 1:
                            lstout.append(opstack.pop())
                    except IndexError:
                        raise SyntaxERROR(key)
                    lstout.append(opstack.pop())
                elif itype[0] == 6:
                    while opstack:
                        lstout.append(opstack.pop())
                    if i[0] == '->':
                        lstout.append(lstin[-1])
                        lstout.append(i)
                        break
                    else:
                        lstout.append(i)
            elif i[0] in self.var:
                lstout.append((self.var[i[0]], i[1]))
            else:
                lstout.append(i)
        while opstack:
            # Don't check if there is a parenthesis or a function.
            # Ignored right parenthesis is allowed.
            lstout.append(opstack.pop())
        return lstout

    def PostEval(self, lstin):
        '''Evaluates the Reverse Polish Expression.'''
        if not lstin:
            return None
        lstsp = []
        lsttemp = []
        for i in lstin:
            if i[0] == ':':
                if lsttemp:
                    lstsp.append(lsttemp)
                else:
                    raise SyntaxERROR(i[1])
                lsttemp = []
            else:
                lsttemp.append(i)
        if lsttemp:
            lstsp.append(lsttemp)
        else:
            raise SyntaxERROR(i[1])
        lstresult = []
        for lst in lstsp:
            numstack = []
            for i in lst:
                oper = op.get(i[0])
                if oper:
                    if oper[0] == 1 and oper[2] != 1:
                        if len(i) == 2:
                            pnum = 1
                        else:
                            pnum = i[2]
                        if isinstance(oper[2], tuple):
                            if pnum not in oper[2]:
                                raise SyntaxERROR(i[1])
                        elif pnum != oper[2]:
                            raise SyntaxERROR(i[1])
                        opnum = pnum
                    else:
                        opnum = oper[2]
                    if len(numstack) >= opnum:
                        argl = []
                        for n in range(opnum):
                            argl.append(numstack.pop()[0])
                        argl.reverse()
                        try:
                            num = self.oeval(i[0], argl)
                            if isinstance(num, (list, tuple)):
                                numstack.append((num, i[1]))
                            else:
                                numstack.append((self.ntype(num), i[1]))
                        except KbdBreak as ex:
                            raise KbdBreak(i[1] + ex.loc + 1)
                        except KeyboardInterrupt:
                            raise KbdBreak(i[1])
                        except MathERROR as ex:
                            raise MathERROR(i[1] + len(i[0]) + ex.loc)
                        except SyntaxERROR as ex:
                            raise SyntaxERROR(i[1] + len(i[0]) + ex.loc + 1)
                        except CannotSolve:
                            raise CannotSolve(i[1])
                        except:
                            raise MathERROR(i[1])
                    else:
                        raise SyntaxERROR(i[1])
                # elif i[0] in self.var:
                #	numstack.append((self.var[i[0]], i[1]))
                else:
                    numstack.append(i)
            if len(numstack) > 1:
                raise SyntaxERROR(numstack[-1][1])
            lstresult.append(numstack[0][0])
        if len(lstresult) == 1:
            return lstresult[0]
        else:
            self.rformat(':', lstresult)
            return lstresult[0]

    def SplitExpr(self):
        '''Split the expression into numbers and operators.'''
        # TODO: Selectable number type.
        tempnum = ''
        pos = 0
        outlist = []
        impmulti = False
        expr = self.expr
        while pos < len(expr):
            i = expr[pos]
            if i in ('"', "'"):
                flen = 1
                tempfx = ''
                while pos + flen < len(expr):
                    j = expr[pos + flen]
                    if j == i:
                        break
                    else:
                        tempfx += j
                    flen += 1
                else:
                    raise SyntaxERROR(pos)
                outlist.append((tempfx, pos))
                pos += flen
            elif ord(i) in range(48, 58):
                tempnum += i
                impmulti = True
            elif i == '.':
                if i in tempnum:
                    raise SyntaxERROR(pos)
                else:
                    tempnum += i
                    impmulti = True
            # TODO: degree m s
            elif i == ' ':
                pass
            else:
                if i in ('E', 'e') and tempnum:
                    # TODO: precise E recognition. Case-sensitive. 4E-5 != 5e-5
                    if i not in tempnum:
                        tempnum += i
                        impmulti = True
                        pos += 1
                        continue
                flen = len(expr) - pos
                oper, var, realop = None, None, None
                while not oper:
                    if flen < 0:
                        break
                    oper = opalias.get(expr[pos:pos + flen])
                    flen -= 1
                if oper:
                    realop = oper
                if not realop:
                    flen = len(expr) - pos
                    while var is None:
                        if flen < 0:
                            break
                        var = self.var.get(expr[pos:pos + flen])
                        flen -= 1
                    if var is not None:
                        # realop=var
                        realop = expr[pos:pos + flen + 1]
                    else:
                        raise SyntaxERROR(pos)
                if tempnum:
                    try:
                        if tempnum[-1] == 'E':
                            outlist.append(
                                (self.ntype(D(tempnum[:-1])), pos - len(tempnum)))
                            outlist.append(('&', pos - len(tempnum) + 1))
                            # outlist.append((self.var['E'],pos))
                            outlist.append(('E', pos))
                        elif tempnum[-1] == 'e':
                            outlist.append(
                                (self.ntype(D(tempnum[:-1])), pos - len(tempnum)))
                            outlist.append(('&', pos - len(tempnum) + 1))
                            outlist.append((self.ntype(c_e), pos))
                        else:
                            outlist.append(
                                (self.ntype(
                                    D(tempnum)),
                                    pos -
                                    len(tempnum)))
                    except:
                        raise SyntaxERROR(pos)
                if oper:
                    if op[oper][0] in (0, 1) and impmulti:
                        # the implicit multiply
                        outlist.append(('&', pos))
                        impmulti = False
                elif var is not None:
                    if impmulti:
                        outlist.append(('&', pos))
                        impmulti = False
                outlist.append((realop, pos))
                if op.get(realop):
                    if op.get(realop)[0] == 5:
                        impmulti = True
                    else:
                        impmulti = False
                else:
                    impmulti = True
                pos += flen
                tempnum = ''
            pos += 1
        pos -= 1
        if tempnum:
            try:
                if tempnum[-1] in ('E', 'e'):
                    outlist.append(
                        (self.ntype(D(tempnum[:-1])), pos - len(tempnum) + 1))
                    outlist.append(('&', pos - len(tempnum) + 2))
                    outlist.append((self.ntype(c_e), pos))
                else:
                    outlist.append(
                        (self.ntype(
                            D(tempnum)),
                            pos -
                            len(tempnum) +
                            1))
            except:
                raise SyntaxERROR(pos)
        lastitem = [1, 2, 1]
        nout = []
        for i in outlist:
            if i[0] in ('+', '-'):
                if lastitem:
                    if lastitem[0] in (1, 3, 4):
                        if i[0] == '-':
                            nout.append(('~', i[1]))
                    else:
                        nout.append(i)
                else:
                    nout.append(i)
            else:
                nout.append(i)
            lastitem = op.get(i[0])
        return nout

    def diff(self, unknown=0, expr=None, varx='X', numtype=None):
        if not expr:
            expr = self.expr
        x = self.ntype(unknown, "decimal")
        mindig = (D(1).next_plus() - D(1)) * 10
        x1 = x - mindig
        x2 = x + mindig
        getcontext().prec += 2
        fx1 = Parser(expr, self.var, 'decimal')
        fx2 = Parser(expr, self.var, 'decimal')
        f1 = fx1.Evaluate(expr, {varx: x1})
        f2 = fx2.Evaluate(expr, {varx: x2})
        while f1 == f2 and mindig != 0:
            if f1 is None:
                if 'err' in fx1.rformat:
                    raise fx1.rformat[2]
                else:
                    raise SyntaxERROR
            elif f2 is None:
                if 'err' in fx2.rformat:
                    raise fx2.rformat[2]
                else:
                    raise SyntaxERROR
            x1 = x - mindig
            x2 = x + mindig
            f1 = fx1.Evaluate(expr, {varx: x1})
            f2 = fx2.Evaluate(expr, {varx: x2})
            mindig = mindig.shift(1)
        if f1 is None:
            if 'err' in fx1.rformat:
                raise fx1.rformat[2]
            else:
                raise SyntaxERROR
        elif f2 is None:
            if 'err' in fx2.rformat:
                raise fx2.rformat[2]
            else:
                raise SyntaxERROR
        result = (f2 - f1) / (x2 - x1)
        getcontext().prec -= 2
        if numtype:
            return self.ntype(result, numtype)
        else:
            return self.ntype(result)

    def newton(self, unknown=0, expr=None, varx='X'):
        sign = lambda x: x and (1, -1)[x < 0]
        if not expr:
            expr = self.expr
        x = self.ntype(unknown, 'decimal')
        func = expr.split('=')
        if len(func) == 1:
            pass
        elif len(func) == 2:
            if func[0] and func[1]:
                expr = "%s-(%s)" % (func[0], func[1])
            else:
                raise SyntaxERROR
        else:
            raise SyntaxERROR
        f1 = Parser(expr, self.var, 'decimal')
        fx = lambda x0: f1.Evaluate(expr, {varx: x0})
        ddx = lambda x0: self.diff(x0, expr, varx, 'decimal')
        minprec = (D(1).next_plus() - D(1)) * 50
        rmax = 100
        count = 1
        ddxx = ddx(x)
        fxx = fx(x)
        sgn = sign(fxx)
        sgnchange = False
        x -= fxx / ddxx
        while abs(fxx) > minprec and count < rmax:
            count += 1
            ddxx = ddx(x)
            if not ddxx:
                x -= sign(x)
                continue
            fxx = fx(x)
            if fxx is None:
                if 'err' in fx1.rformat:
                    x -= sign(x)
                    continue
                else:
                    raise SyntaxERROR
            if (not sgnchange) and sgn != sign(fxx):
                sgnchange = True
            x -= fxx / ddxx
        if count >= rmax and abs(fxx) > minprec:
            if sgnchange:
                rmax = 10000
            else:
                rmax = 1000
                x = -x
            while abs(fxx) > minprec and count < rmax:
                count += 1
                ddxx = ddx(x)
                if not ddxx:
                    x -= sign(x)
                    continue
                fxx = fx(x)
                if fxx is None:
                    if 'err' in fx1.rformat:
                        x -= sign(x)
                        continue
                    else:
                        raise SyntaxERROR
                x -= fxx / ddxx
            if count >= rmax and abs(fxx) > minprec:
                raise CannotSolve
        return self.ntype(x)

    def brent(self, unka=-100, unkb=100, expr=None, varx='X'):
        # "Comments on An Improvement to the Brent's Method"
        if not expr:
            expr = self.expr
        a = self.ntype(unka, 'decimal')
        b = self.ntype(unkb, 'decimal')
        minprec = (D(1).next_plus() - D(1)) * 50
        s = 0
        func = expr.split('=')
        if len(func) == 1:
            pass
        elif len(func) == 2:
            if func[0] and func[1]:
                expr = "%s-(%s)" % (func[0], func[1])
            else:
                raise SyntaxERROR
        else:
            raise SyntaxERROR
        func = Parser(expr, self.var, 'decimal')
        f = lambda x0: func.Evaluate(expr, {varx: x0})
        if b < a:
            a, b = b, a
        fa = f(a)
        fb = f(b)
        # fs = 0
        if fa * fb > 0:
            raise MathERROR
        elif abs(fa) <= minprec:
            return self.ntype(unka)
        elif abs(fb) <= minprec:
            return self.ntype(unkb)
        count = 1
        rmax = 1000
        while not (abs(fa) <= minprec or abs(fb) <= minprec or abs(
                b - a) <= minprec or count > rmax):
            c = (a + b) / 2
            fc = f(c)
            if (fa != fc) and (fb != fc):
                # Inverse quadratic interpolation
                s = a * fb * fc / (fa - fb) / (fa - fc) + b * fa * fc / \
                    (fb - fa) / (fb - fc) + c * fa * fb / (fc - fa) / (fc - fb)
                if a < s < b:
                    pass  # fs = f(s)
                else:
                    s = c
            else:
                # Secant Rule
                s = b - fb * (b - a) / (fb - fa)
                # fs = f(s)
            if c > s:
                s, c = c, s
            if f(c) * f(s) <= 0:
                a, b = c, s
            else:
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = s
            fa = f(a)
            fb = f(b)
            count += 1
        if count > rmax:
            raise CannotSolve
        elif fa <= minprec:
            return self.ntype(a)
        else:
            return self.ntype(b)

    def oeval(self, o, a):
        # should be type-independent
        N = self.ntype
        if o in self.var:
            return self.var[o]
        elif o == "Rand":
            return random.random()
        elif o == "i":
            return cfrac(0, 1)
        elif o == "pi":
            return c_pi  # pi()
        elif o == "e":
            return c_e  # D(1).exp()
        elif o == "_":
            return self.var["Ans"]
        elif o == "(":
            return a[0]
        elif o == "Pol(":
            # cfrac
            x = cmath.polar(complex(a[0], a[1]))
            self.var['X'] = N(x[0])
            self.var['Y'] = N(x[1])
            self.rformat = ('Pol', x)
            return x[0]
        elif o == "Rec(":
            # cfrac
            x = cmath.rect(a)
            self.var['X'] = N(x.real)
            self.var['Y'] = N(x.imag)
            self.rformat = ('Rec', (x.real, x.imag))
            return x.real
        elif o == "integr(":
            # for i in frange()
            a0 = N(a[1], "decimal")
            b0 = N(a[2], "decimal")
            f = 0
            fx = lambda x0: Parser(
                a[0], self.var, 'decimal').Evaluate(
                a[0], {
                    'X': x0})
            delta = (b0 - a0) / 10**(4 + (b0 - a0).log10())
            lastx = a0
            for x in frange(a0, b0, delta):
                f += (x - lastx) / 6 * \
                    (fx(lastx) + 4 * fx((lastx + x) / 2) + fx(x))
                lastx = x
            f += (b0 - lastx) / 6 * \
                (fx(lastx) + 4 * fx((b0 + lastx) / 2) + fx(b0))
            return N(f)
        elif o == "d/dx(":
            return self.diff(a[1], a[0], "X")
        elif o == "Sigma(":
            if a[2] < a[1]:
                raise ArgumentERROR
            x1 = N(a[1], 'int')
            x2 = N(a[2], 'int')
            if a[1] != x1 or a[2] != x2:
                raise ArgumentERROR
            f = 0
            for x in range(x1, x2 + 1):
                f += Parser(a[0], self.var, 'decimal').Evaluate(a[0], {'X': x})
                # print((f,))
            return N(f)
        elif o == "Pi(":
            if a[2] < a[1]:
                raise ArgumentERROR
            x1 = N(a[1], 'int')
            x2 = N(a[2], 'int')
            if a[1] != x1 or a[2] != x2:
                raise ArgumentERROR
            f = 1
            for x in range(x1, x2 + 1):
                f *= Parser(a[0], self.var, 'decimal').Evaluate(a[0], {'X': x})
            return N(f)
        elif o == "solve(":
            if len(a) == 1:
                x = a[0]
                try:
                    x = self.brent(-
                                   10**(getcontext().prec -
                                        2), 10**(getcontext().prec -
                                                 2), a[0])
                except (KbdBreak, KeyboardInterrupt):
                    raise KbdBreak
                except:
                    try:
                        x = self.newton(0, a[0])
                    except (KbdBreak, KeyboardInterrupt):
                        raise KbdBreak
                    except:
                        raise CannotSolve
                self.rformat = ('Solve', (x, 'X'))
                self.var['X'] = x
                return x
            elif len(a) == 2:
                x = self.newton(a[1], a[0])
                self.rformat = ('Solve', (x, 'X'))
                self.var['X'] = x
                return x
            elif len(a) == 3:
                x = self.brent(a[1], a[2], a[0])
                self.rformat = ('Solve', (x, 'X'))
                self.var['X'] = x
                return x
            elif len(a) == 4:
                x = self.brent(a[1], a[2], a[0], a[3])
                self.rformat = ('Solve', (x, a[2]))
                self.var[a[2]] = x
                return x
        elif o == "nsolve(":
            if len(a) == 2:
                x = self.newton(a[1], a[0])
                self.rformat = ('Solve', (x, 'X'))
                self.var['X'] = x
                return x
            else:
                x = self.newton(a[1], a[0], a[2])
                self.rformat = ('Solve', (x, a[2]))
                self.var[a[2]] = x
                return x
        elif o == "P(":
            pass
        elif o == "Q(":
            pass
        elif o == "R(":
            pass
        elif o == "Mod(":
            return a[0] % a[1]
        elif o == "sin(":
            return sin(N(a[0], 'decimal'))
        elif o == "cos(":
            return cos(N(a[0], 'decimal'))
        elif o == "tan(":
            return sin(N(a[0], 'decimal')) / cos(N(a[0], 'decimal'))
        elif o == "cot(":
            return cos(N(a[0], 'decimal')) / sin(N(a[0], 'decimal'))
        elif o == "sec(":
            return 1 / cos(N(a[0], 'decimal'))
        elif o == "csc(":
            return 1 / sin(N(a[0], 'decimal'))
        elif o == "asin(":
            # TODO: use frac instead of float
            return math.asin(a[0])
        elif o == "acos(":
            return math.acos(a[0])
        elif o == "atan(":
            return math.atan(a[0])
        elif o == "sinh(":
            return sinh(N(a[0], 'decimal'))
        elif o == "cosh(":
            return cosh(N(a[0], 'decimal'))
        elif o == "tanh(":
            return sinh(N(a[0], 'decimal')) / cosh(N(a[0], 'decimal'))
        elif o == "asinh(":
            return math.asinh(a[0])
        elif o == "acosh(":
            return math.acosh(a[0])
        elif o == "atanh(":
            return math.atanh(a[0])
        elif o == "log(":
            if len(a) == 1:
                return N(a[0], 'decimal').log10()
            else:
                return N(a[0], 'decimal').ln() / N(a[0], 'decimal').ln()
        elif o == "ln(":
            return N(a[0], 'decimal').ln()
        elif o == "exp(":
            return N(a[0], 'decimal').exp()
        elif o == "sqrt(":
            return a[0]**N('0.5')
        elif o == "cbrt(":
            return a[0]**(N(1) / N(3))
        elif o == "Arg(":
            # cmath.phase(x)
            pass
        elif o == "Abs(":
            return abs(a[0])
        elif o == "Conjg(":
            return cfrac(a[0]).conjugate()
        elif o == "Real(":
            return a[0].real
        elif o == "Imag(":
            return a[0].imag
        elif o == "Not(":
            return ~a[0]
        elif o == "Neg(":
            return -a[0]
        elif o == "Det(":
            return det(a[0])
        elif o == "Trn(":
            return zip(*a[0])
        elif o == "Ide(":
            x = [[0 for x in range(a[0])] for x in range(a[0])]
            for i in range(a[0]):
                x[i][i] = 1
            return x
        elif o == "Adj(":
            pass
        elif o == "Inv(":
            pass
        elif o == "Round(":
            # TODO: Norm, Fix, Sci Mode, Complex
            return N(str(a[0]), 'decimal')
        elif o == "int(":
            return N(a[0], 'int')
        elif o == "factor(":
            if a[0] > 0 and int(a[0]) == a[0] and a[
                    0] < 10**(getcontext().prec - 1):
                self.rformat = ('fact', pfact(N(a[0], 'int')))
                return N(a[0], 'int')
            else:
                raise MathERROR
        elif o == "LCM(":
            return lcm(*a)
        elif o == "GCD(":
            return gcd(*a)
        elif o == "Q...r(":
            x = divmod(a[0], a[1])
            self.var['C'] = x[0]
            self.var['D'] = x[1]
            self.rformat = ('Qr', x)
            return x[0]
        elif o == "i~Rand(":
            return random.randrange(a[0], a[1])
        elif o == "table(":
            tbl = []
            if len(a) == 2:
                a = (a[0], -abs(a[1]), abs(a[1]), 1)
            elif len(a) == 3:
                a = (a[0], a[1], a[2], 1)
            # TODO: multi func "f(x); g(x)"
            for i in frange(a[1], a[2], a[3]):
                x = Parser(a[0], self.var, 'decimal').Evaluate(a[0], {'X': i})
                tbl.append((i, x))
            self.rformat = ('table', (2, tbl))
            return tbl
        elif o == "^":
            return a[0] ** a[1]
        elif o == "!":
            return math.factorial(int(a[0]))
        elif o == "`":
            # degree minute second
            pass
        elif o == "(deg)":
            # TODO: Deg, Rad, Gra Mode
            return a[0] / 180 * c_pi
        elif o == "(rad)":
            return a[0]
        elif o == "(gra)":
            return a[0] / 200 * c_pi
        elif o == "xroot":
            return a[1]**(1 / N(a[0]))
        elif o == ">t":
            pass
        elif o == "%":
            return a[0] / 100
        elif o == "\\":
            # TODO: explicit fractions
            pass
        elif o == "~":
            # "~" == "(-)" on fx-XX
            return -a[0]
        elif o == "d":
            # TODO: BASE
            return a[0]
        elif o == "h":
            return int(str(a[0]), 16)
        elif o == "b":
            return int(str(a[0]), 2)
        elif o == "o":
            return int(str(a[0]), 8)
        elif o == "~x":
            # TODO: STAT
            pass
        elif o == "~y":
            pass
        elif o == "~x1":
            pass
        elif o == "~x2":
            pass
        elif o == "&":
            # "&" == (implicit) "*"
            return a[0] * a[1]
        elif o == "nPr":
            return math.factorial(
                int(a[0])) // math.factorial(int(a[0] - a[1]))
        elif o == "nCr":
            return math.factorial(
                int(a[0])) // math.factorial(int(a[1])) // math.factorial(int(a[0] - a[1]))
        elif o == "<":
            # TODO: use frac instead of float
            return cmath.rect(r, phi)
        elif o == "@":
            return sum(p * q for p, q in zip(a[0], a[1]))
        elif o == "*":
            # TODO: vectors and matrics
            return a[0] * a[1]
        elif o == "/":
            return a[0] / a[1]
        elif o == "+":
            # TODO: vectors and matrics
            return a[0] + a[1]
        elif o == "-":
            return a[0] - a[1]
        elif o == "and":
            return int(a[0]) & int(a[1])
        elif o == "or":
            return int(a[0]) | int(a[1])
        elif o == "xor":
            return int(a[0]) ^ int(a[1])
        elif o == "xnor":
            return ~(int(a[0]) ^ int(a[1]))
        elif o == "=":
            # TODO: CALC and SOLVE
            pass
        elif o == ")":
            return a[0]
        elif o == "M+":
            self.var['M'] += a[0]
            return a[0]
        elif o == "M-":
            self.var['M'] -= a[0]
            return a[0]
        elif o == "->":
            self.var[a[1]] = a[0]
            return a[0]
        elif o == ">rtheta":
            x = cmath.polar(complex(a[0]))
            self.rformat = ('Rtheta', x)
            return a[0]
        elif o == ">a+bi":
            x = cmath.rect(a)
            self.rformat = ('ABi', x)
            return a[0]
        elif o == ":":
            pass
        else:
            raise SyntaxERROR

    def Evaluate(self, expr=None, var={}):
        self.result = None
        self.rformat = None
        self.var.update(var)
        if expr == '':
            return
        if expr:
            self.expr = str(expr).strip()
        if not self.expr:
            return
        # TODO: special expressions as functions
        elif self.expr.lower() == 'help':
            # rewrite this
            # help:xxx
            rstr = 'Functions and Operators:\n'
            count = 0
            for i in op:
                count += 1
                if count % 4:
                    rstr += i + '\t'
                else:
                    rstr += i + '\n'
            self.rformat = ('info', rstr)
            return None
        elif self.expr.lower() == "eqn":
            # TODO: eqn mode
            pass
            # TODO: mode, settings
            # setting name
        elif self.expr == '=':
            return self.var['Ans']
        elif self.expr.lower() == 'dms':
            self.result = self.var['Ans']
            self.rformat = ('dms', (self.var['Ans'], 0, 0))
            return self.var['Ans']
            # solve('(24-X)*2pi/X*365.256=2pi',22,25
        elif self.expr.lower() == 'fact':
            calc = Parser(
                "factor(%s)" % str(
                    self.var['Ans']),
                self.var,
                'decimal')
            calc.Evaluate()
            self.rformat = calc.rformat
            return self.var['Ans']
        elif self.expr.lower() == 'decimal':
            self.numtype = 'decimal'
            self.rformat = ('info', 'Set to decimal number mode.')
        elif self.expr.lower() == 'real':
            self.numtype = 'frac'
            self.rformat = ('info', 'Set to real number mode.')
        elif self.expr.lower() == 'complex':
            self.numtype = 'cfrac'
            self.rformat = ('info', 'Set to complex number mode.')
        else:
            try:
                postlist = self.SplitExpr()
                try:
                    # TODO: Multi result
                    self.result = self.PostEval(self.In2Post(postlist))
                    self.var['Ans'] = self.result
                    return self.result
                    # TODO: better result rformat, print or report
                except MathERROR as ex:
                    self.rformat = (
                        'err', "Math ERROR:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                    # raise MathERROR
                except SyntaxERROR as ex:
                    self.rformat = (
                        'err', "Syntax ERROR:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                    # raise SyntaxERROR
                except KbdBreak as ex:
                    self.rformat = (
                        'err', "Keyboard Break:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                    # raise KbdBreak
                except DimensionERROR as ex:
                    self.rformat = (
                        'err', "Dimension ERROR:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                except CannotSolve as ex:
                    self.rformat = (
                        'err', "Can't Solve:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                except VariableERROR as ex:
                    self.rformat = (
                        'err', "Variable ERROR:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                except TimeOut as ex:
                    self.rformat = (
                        'err', "Time Out:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
                except ArgumentERROR as ex:
                    self.rformat = (
                        'err', "Argument ERROR:\n %s\n %s" %
                        (self.expr, ' ' * ex.loc + '^'), ex)
                    self.result = None
            except SyntaxERROR as ex:
                self.rformat = (
                    'err', "Syntax ERROR:\n %s\n %s" %
                    (self.expr, ' ' * ex.loc + '^'), ex)
                self.result = None
                # raise SyntaxERROR
            return self.result

    def PrintResult(self):
        # TODO: matrix, vct
        # TODO: math and line
        N = self.ntype
        if self.rformat:
            f = self.rformat[0]
            r = self.rformat[1]
            if f == 'info':
                return r
            elif f == 'err':
                return r
            elif f == 'Pol' and self.result == r[0]:
                return 'r=%s\nθ=%s' % (
                    str(N(r[0], 'decimal')), str(N(r[1], 'decimal')))
            elif f == 'Rec' and self.result == r[0]:
                return 'X=%s\nY=%s' % (
                    str(N(r[0], 'decimal')), str(N(r[1], 'decimal')))
            elif f == 'Qr' and self.result == r[0]:
                return 'Q=%s\nR=%s' % (
                    str(N(r[0], 'decimal')), str(N(r[1], 'decimal')))
            elif f == 'Solve' and self.result == r[0]:
                return '%s=%s' % (r[1], str(N(r[0], 'decimal')))
            elif f == 'dms':
                x = self.result
                self.rformat = (
                    'dms', (int(x), int(
                        (x * 60) %
                        60), (x * 3600) %
                        60))
                r = self.rformat[1]
                return "%s°%s’%s”" % (
                    str(N(r[0], 'decimal')), str(N(r[1], 'decimal')), str(N(r[2], 'decimal')))
            elif f == 'fact':
                factor, count = r[0], 0
                fl = []
                for i in r:
                    if i == factor:
                        count += 1
                    else:
                        if count == 1:
                            fl.append(str(factor))
                        else:
                            fl.append(
                                '(' + str(factor) + '^' + str(count) + ')')
                        factor = i
                        count = 1
                if count == 1:
                    fl.append(str(factor))
                else:
                    fl.append('(' + str(factor) + '^' + str(count) + ')')
                return '*'.join(i for i in fl)
            elif f == 'table':
                pass
                return ""
        elif isinstance(self.result, (str, list, tuple)):
            return str(self.result)
        elif self.expr == '=':
            return str(N(self.var['Ans'], 'decimal'))
        elif self.numtype == 'decimal':
            return str(N(self.result, 'decimal'))
        elif self.numtype == 'frac':
            if self.result == 0 or abs(N(self.result, 'float')) > 1e6:
                return frac(self.result).pretty()
            piratio = frac(self.result / c_pi)
            if piratio.t == 2:
                return frac(self.result).pretty()
            else:
                rstr = piratio.pretty()
                rl = rstr.split()
                if not piratio:
                    return frac(self.result).pretty()
                # TODO: use before after
                if piratio == 1:
                    return 'π'
                elif piratio == -1:
                    return '-π'
                elif len(rl) in (1, 2):
                    return rstr + 'π'
                elif len(rl) == 3:
                    rl[1] += ' π'
                    return '\n'.join(l for l in rl)
                elif len(rl) == 4:
                    rl[2] += ' π'
                    return '\n'.join(l for l in rl)
        elif self.numtype == 'cfrac':
            return cfrac(self.result).pretty()
        elif self.numtype == 'float':
            return str(float(self.result))
        elif self.numtype == 'int':
            return str(int(self.result))
        elif self.result is None:
            return ''


def main():
    fx991es = Parser()
    print(
        "== F-Calc %s ==\nuse the LineIO expressions of Casio fx-???ES/MS or Canon f-7??SGA for calculation." %
        __version__)
    a = input("> ")
    while True:
        r = fx991es.Evaluate(a)
        if r is not None or fx991es.rformat:
            print(fx991es.PrintResult())
        try:
            a = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
    print("Bye.")
    return 0

if __name__ == '__main__':
    main()
