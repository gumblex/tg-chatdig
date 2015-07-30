#！/usr/bin/env python3
# -*- coding： utf-8 -*-

import sys, os
import re
import jieba
#from zhconv import convert_for_mw
from zhutil import *

punctstr = (
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~¢£¥·ˇˉ―‖‘’“”•′‵、。々'
    '〈〉《》「」『』【】〔〕〖〗〝〞︰︱︳︴︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄'
    '﹏﹐﹒﹔﹕﹖﹗﹙﹚﹛﹜﹝﹞！（），．：；？［｛｜｝～､￠￡￥')


ucjk = frozenset(itertools.chain(
    range(0x1100, 0x11FF + 1),
    range(0x2E80, 0xA4CF + 1),
    range(0xA840, 0xA87F + 1),
    range(0xAC00, 0xD7AF + 1),
    range(0xF900, 0xFAFF + 1),
    range(0xFE30, 0xFE4F + 1),
    range(0xFF65, 0xFFDC + 1),
    range(0xFF01, 0xFF0F + 1),
    range(0xFF1A, 0xFF20 + 1),
    range(0xFF3B, 0xFF40 + 1),
    range(0xFF5B, 0xFF60 + 1),
    range(0x1F000, 0x2FFFF + 1)
))

RE_BRACKET = re.compile(' ?[（(][^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U0001F000-\U0001F8AD\U00020000-\U0002A6D6)）]*[)）]|"[^"]+"')

brackets = '()（）[]""‘’“”{}〈〉《》「」『』【】〔〕〖〗'

_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__), path))

jiebazhc = jieba.Tokenizer(_get_module_path('zhcdict.txt'))
jiebazhc.cache_file = "jiebazhc.cache"

#RE_BRACKETS = re.compile(' ?\((.*?)\)| ?\((.*?)\)')
RE_BRACKETS = re.compile('|'.join(' ?%s.*?%s' % (re.escape(brackets[i]), re.escape(brackets[i+1])) for i in range(0, len(brackets), 2)))

tailp = frozenset("""([{£¥`〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛"'“‘""")
stripblank = lambda s: s.replace(' ', '').replace('\u3000', '')

if len(sys.argv) > 1:
	if sys.argv[1] == 'noop':
		cut = lambda s: (s,)
		stripblank = lambda s: s.replace('\u3000', ' ')
	else:
		cut = lambda s: jiebazhc.cut(s, HMM=False)
else:
	cut = lambda s: jieba.cut(s, HMM=False)

notchinese = lambda l: not l or sum((ord(i) not in ucjk) for i in l) > .5 * len(l)
brcksub = lambda matchobj: '' if notchinese(matchobj.group(0)[1:-1]) else matchobj.group(0)

def cutandsplit(s):
	for ln in filterlist(splitsentence(stripblank(s))):
		l = RE_BRACKETS.sub(brcksub, ln.strip())
		if notchinese(l):
			continue
		yield ' '.join(cut(l.replace('「', '“').replace('」', '”').replace('『', '‘').replace('』', '’').lstrip(tailpunct).rstrip(headpunct)))

cutfilter = lambda s: ' '.join(i.strip() for i in cut(s.replace(' ', '')))

lastline = ''

for ln in sys.stdin:
	l = ln.strip(' \t\n\r\x0b\x0c\u3000=[]')
	if not l or all((ord(i) not in ucjk) for i in l) or any((ord(i) in range(32)) for i in l):
		continue
	elif l[-1] in tailp:
		lastline += l
	else:
		#sys.stdout.write('\n'.join(filterlist((splitsentence(cutfilter(lastline + l))))) + '\n')
		sys.stdout.write('\n'.join(cutandsplit(lastline + l)))
		sys.stdout.write('\n')
		lastline = ''

if lastline:
	#sys.stdout.write('\n'.join(filterlist((splitsentence(cutfilter(lastline))))) + '\n')
	sys.stdout.write('\n'.join(cutandsplit(lastline)))
	sys.stdout.write('\n')
