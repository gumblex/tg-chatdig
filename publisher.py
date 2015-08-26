#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import sqlite3
import operator
import collections

import jinja2
import truecaser
import jieba.analyse

TIMEZONE = 8 * 3600

CFG = json.load(open('config.json'))
db = sqlite3.connect('chatlog.db')
conn = db.cursor()

USER_CACHE = {}

re_word = re.compile(r"\w+", re.UNICODE)
re_tag = re.compile(r"#\w+", re.UNICODE)
re_at = re.compile('@[A-Za-z][A-Za-z0-9_]{4,}')
re_url = re.compile(r"(^|[\s.:;?\-\]<\(])(https?://[-\w;/?:@&=+$\|\_.!~*\|'()\[\]%#,]+[\w/#](\(\))?)(?=$|[\s',\|\(\).:;?\-\[\]>\)])")
_ig1 = operator.itemgetter(1)

MEDIA_TYPES = {
'text': '文本',
'audio': '声音',
'document': '文件',
'photo': '图片',
'sticker': '贴纸',
'video': '视频',
'voice': '语音',
'contact': '名片',
'location': '位置',
'service': '服务'
}

SERVICE = frozenset(('new_chat_participant', 'left_chat_participant', 'new_chat_title', 'new_chat_photo', 'delete_chat_photo', 'group_chat_created'))

def daystart(sec=None):
    if not sec:
        sec = time.time()
    return int((sec + TIMEZONE) // 86400 * 86400 - TIMEZONE)

def uniq(seq, key=None): # Dave Kirby
    # Order preserving
    seen = set()
    if key:
        return [x for x in seq if key(x) not in seen and not seen.add(key(x))]
    else:
        return [x for x in seq if x not in seen and not seen.add(x)]

def db_getuser(uid):
    r = USER_CACHE.get(uid)
    if r is None:
        r = conn.execute('SELECT username, first_name, last_name FROM users WHERE id = ?', (uid,)).fetchone() or (None, None, None)
        USER_CACHE[uid] = r
    return r

def db_isbot(uid):
    return (db_getuser(uid)[0] or '').lower().endswith('bot')

def db_getufname(uid, mmedia=None):
    if uid == CFG['ircbotid']:
        if mmedia and '_ircuser' in mmedia:
            name = mmedia['_ircuser']
        else:
            name = '<IRC 用户>'
    else:
        name, last = db_getuser(uid)[1:]
        if last:
            name += ' ' + last
    if len(name) > 50:
        name = name[:50] + '[…]'
    return name

def db_getfirstname(uid, mmedia=None):
    if uid == CFG['ircbotid']:
        if mmedia and '_ircuser' in mmedia:
            return mmedia['_ircuser']
        else:
            return '<IRC 用户>'
    else:
        return db_getufname(uid).split()[0]

def strftime(fmt, t=None):
    if t is None:
        t = time.time()
    t += TIMEZONE
    return time.strftime(fmt, time.gmtime(t))

class StatComposer:

    def __init__(self):
        self.template = 'stat.html'
        self.tc = truecaser.Truecaser(truecaser.loaddict(open('vendor/truecase.txt', 'rb')))

    def fetchmsgstat(self):
        self.msglen = self.start = self.end = 0
        mediactr = collections.Counter()
        usrctr = collections.Counter()
        tags = collections.Counter()
        for mid, src, text, date, media in conn.execute('SELECT id, src, text, date, media FROM messages ORDER BY date ASC, id ASC'):
            text = text or ''
            if not self.start:
                self.start = date
            self.start = min(self.start, date)
            self.end = max(self.end, date)
            for tag in re_tag.findall(text):
                tags[self.tc.truecase(tag)] += 1
            media = json.loads(media or '{}')
            if media.get('type') in MEDIA_TYPES:
                t = media['type']
            else:
                mt = media.keys() & MEDIA_TYPES.keys()
                if mt:
                    t = tuple(mt)[0]
                elif media.keys() & SERVICE:
                    t = 'service'
                else:
                    t = 'text'
            mediactr[t] += 1
            usrctr[src] += 1
            self.msglen += 1
        self.end = date
        types = [(MEDIA_TYPES[k], v) for k, v in mediactr.most_common()]
        tags = sorted(filter(lambda x: x[1] > 2, tags.items()), key=lambda x: -x[1])
        return types, tags, usrctr

    def generalinfo(self):
        types, tags, usrctr = self.fetchmsgstat()
        mcomm = usrctr.most_common()
        count = self.msglen
        stat = {
            'start': strftime('%Y-%m-%d %H:%M:%S', self.start),
            'end': strftime('%Y-%m-%d %H:%M:%S', self.end),
            'count': count,
            'freq': '%.2f' % (count * 60 / (self.end - self.start)),
            'flooder': tuple(((k, db_getufname(k)), db_getuser(k)[0] or '', v, '%.2f%%' % (v/count*100)) for k, v in mcomm),
            'types': types,
            'tags': tags,
            'avg': '%.2f' % (count / len(usrctr))
        }
        return stat

    def render(self):
        kvars = {'info': self.generalinfo()}
        template = jinja2.Environment(loader=jinja2.FileSystemLoader('templates')).get_template(self.template)
        return template.render(**kvars)

class DigestManager:

    def __init__(self):
        self.template = 'stat.html'
        self.tc = truecaser.Truecaser(truecaser.loaddict(open('vendor/truecase.txt', 'rb')))
        self.stopwords = frozenset(map(str.strip, open('vendor/stopwords.txt', 'r', encoding='utf-8')))
        self.ircbots = re.compile(r'(titlbot|varia|Akarin).*')
        self.fetchmsg()



start = time.time()

sc = StatComposer()
print(sc.render())
sys.stderr.write('Done in %.4gs.\n' % (time.time() - start))
