#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import math
import time
import json
import queue
import random
import logging
import sqlite3
import threading
import functools
import subprocess
import collections

import requests
from vendor import libirc
from vendor import zhutil
from vendor import zhconv
from vendor import simpleime
from vendor import mosesproxy
from vendor import chinesename
#from vendor import fparser

__version__ = '1.0'

# (昨日)
# 今日焦点: xx,yy,zz (12345,45678)
# (今日标签: #xx,#yy)
# (今日语录: ......)

#jieba.re_eng = re.compile('[a-zA-Z0-9_]', re.U)

MEDIA_TYPES = frozenset(('audio', 'document', 'photo', 'sticker', 'video', 'contact', 'location', 'new_chat_participant', 'left_chat_participant', 'new_chat_title', 'new_chat_photo', 'delete_chat_photo', 'group_chat_created', '_ircuser'))

logging.basicConfig(stream=sys.stdout, format='# %(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

HSession = requests.Session()
USERAGENT = 'TgChatDiggerBot/%s %s' % (__version__, HSession.headers["User-Agent"])
HSession.headers["User-Agent"] = USERAGENT

db = sqlite3.connect('chatlog.db')
conn = db.cursor()
conn.execute('''CREATE TABLE IF NOT EXISTS messages (
id INTEGER PRIMARY KEY,
src INTEGER,
text TEXT,
media TEXT,
date INTEGER,
fwd_src INTEGER,
fwd_date INTEGER,
reply_id INTEGER
)''')
conn.execute('''CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY,
username TEXT,
first_name TEXT,
last_name TEXT
)''')
conn.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, val INTEGER)')
# conn.execute('CREATE TABLE IF NOT EXISTS words (word TEXT PRIMARY KEY, count INTEGER)')

class LRUCache:

    def __init__(self, maxlen):
        self.capacity = maxlen
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def get(self, key, default=None):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return default

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

### Polling

def getupdates():
    global OFFSET, MSG_Q
    while 1:
        try:
            updates = bot_api('getUpdates', offset=OFFSET)
        except Exception as ex:
            logging.exception('Get updates failed.')
            continue
        if updates:
            logging.debug('Messages coming.')
            OFFSET = updates[-1]["update_id"] + 1
            for upd in updates:
                MSG_Q.put(upd)
        time.sleep(.1)

def getsaying():
    global SAY_P, SAY_Q
    while 1:
        say = getsayingbytext()
        SAY_Q.put(say)

def getsayingbytext(text=''):
    global SAY_P
    with SAY_LCK:
        try:
            SAY_P.stdin.write(text.strip().encode('utf-8') + b'\n')
            SAY_P.stdin.flush()
            say = SAY_P.stdout.readline().strip().decode('utf-8')
        except BrokenPipeError:
            SAY_P = subprocess.Popen(SAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='vendor')
            SAY_P.stdin.write(text.strip().encode('utf-8') + b'\n')
            SAY_P.stdin.flush()
            say = SAY_P.stdout.readline().strip().decode('utf-8')
    return say

def geteval(text=''):
    global EVIL_P
    with EVIL_LCK:
        if EVIL_P.returncode is not None:
            EVIL_P = subprocess.Popen(EVIL_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='vendor')
        try:
            result, errs = EVIL_P.communicate(text.strip().encode('utf-8'), timeout=10)
        except Exception: # TimeoutExpired
            EVIL_P.kill()
            result, errs = EVIL_P.communicate()
        result = result.strip().decode('utf-8', errors='replace')
    return result

def checkircconn():
    global ircconn
    if not ircconn or not ircconn.sock:
        ircconn = libirc.IRCConnection()
        ircconn.connect((CFG['ircserver'], CFG['ircport']), use_ssl=CFG['ircssl'])
        ircconn.setnick(CFG['ircnick'])
        ircconn.setuser(CFG['ircnick'], CFG['ircnick'])
        ircconn.join(CFG['ircchannel'])
        logging.info('IRC (re)connected.')

def getircupd():
    global MSG_Q, IRCOFFSET
    while 1:
        checkircconn()
        line = ircconn.parse(block=False)
        if line and line["cmd"] == "PRIVMSG":
            if line["dest"] != CFG['ircnick'] and not re.match(CFG['ircbanre'], line["nick"]):
                msg = {
                    'message_id': IRCOFFSET,
                    'from': {'id': CFG['ircbotid'], 'first_name': CFG['ircbotname'], 'username': 'orzirc_bot'},
                    'date': int(time.time()),
                    'chat': {'id': -CFG['groupid'], 'title': CFG['ircchannel']},
                    'text': line["msg"].strip(),
                    '_ircuser': line["nick"]
                }
                MSG_Q.put({'update_id': IRCOFFSET, 'message': msg})
                IRCOFFSET += 1
        time.sleep(.5)

def irc_send(text='', reply_to_message_id=None, forward_message_id=None):
    if ircconn:
        checkircconn()
        if reply_to_message_id:
            m = MSG_CACHE.get(reply_to_message_id, {})
            if 'from' in m:
                text = "%s: %s" % (db_getufname(m['from']['id']), text)
        elif forward_message_id:
            m = db_getmsg(forward_message_id)
            if m:
                text = "Fwd %s: %s" % (db_getufname(m[1]), m[2])
        text = text.strip()
        if text.count('\n') < 2:
            ircconn.say(CFG['ircchannel'], text)

### DB import

def importdb(filename):
    logging.info('Import DB...')
    if not os.path.isfile(filename):
        logging.warning('DB not found.')
        return
    db_s = sqlite3.connect(filename)
    conn_s = db_s.cursor()
    for vals in conn_s.execute('SELECT id, src, text, media, date, fwd_src, fwd_date, reply_id FROM messages WHERE dest = ?', (CFG['groupid'],)):
        vals = list(vals)
        vals[0] = -250000 + vals[0]
        conn.execute('INSERT OR IGNORE INTO messages (id, src, text, media, date, fwd_src, fwd_date, reply_id) VALUES (?,?,?,?, ?,?,?,?)', vals)
    for vals in conn_s.execute('SELECT id, username, first_name, last_name FROM users'):
        conn.execute('INSERT OR IGNORE INTO users (id, username, first_name, last_name) VALUES (?,?,?,?)', vals)
    db.commit()
    logging.info('DB import done.')

def importupdates(offset, number=5000):
    off = OFFSET - number
    updates = bot_api('getUpdates', offset=off, limit=100)
    while updates:
        logging.info('Imported %s - %s' % (off, updates[-1]["update_id"]))
        off = updates[-1]["update_id"] + 1
        for d in updates:
            uid = d['update_id']
            if 'message' in d:
                msg = d['message']
                cls = classify(msg)
                if cls == 0 and msg['chat']['id'] == -CFG['groupid']:
                    logmsg(msg, True)
                elif cls == 1:
                    logmsg(msg, True)
        time.sleep(.1)
        updates = bot_api('getUpdates', offset=off, limit=100)

### API Related

class BotAPIFailed(Exception):
    pass

def change_session():
    global HSession
    HSession.close()
    HSession = requests.Session()
    HSession.headers["User-Agent"] = USERAGENT
    logging.warning('Session changed.')

def bot_api(method, **params):
    for att in range(2):
        try:
            req = HSession.get(URL + method, params=params)
            retjson = req.content
            ret = json.loads(retjson.decode('utf-8'))
            break
        except Exception as ex:
            if att < 1:
                change_session()
            else:
                raise ex
    if not ret['ok']:
        raise BotAPIFailed(repr(ret))
    return ret['result']

def bot_api_noerr(method, **params):
    try:
        bot_api(method, **params)
    except Exception:
        logging.exception('Async bot API failed.')

def async_send(method, **params):
    threading.Thread(target=bot_api_noerr, args=(method,), kwargs=params).run()

def sendmsg(text, chat_id, reply_to_message_id=None):
    logging.info('sendMessage(%s): %s' % (len(text), text[:20]))
    if len(text) > 2000:
        text = text[:1999] + '…'
    if reply_to_message_id and reply_to_message_id < 0:
        reply_to_message_id = None
    m = bot_api('sendMessage', chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id)
    if chat_id == -CFG['groupid']:
        logmsg(m)
        irc_send(text, reply_to_message_id=reply_to_message_id)

def forward(message_id, chat_id, reply_to_message_id=None):
    logging.info('forwardMessage: %r' % message_id)
    try:
        r = bot_api('forwardMessage', chat_id=chat_id, from_chat_id=-CFG['groupid'], message_id=message_id)
        logging.debug('Forwarded: %s' % message_id)
    except BotAPIFailed as ex:
        m = db_getmsg(message_id)
        if m:
            r = sendmsg('[%s] %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(m[4] + CFG['timezone'] * 3600)), db_getufname(m[1]), m[2]), chat_id, reply_to_message_id)
            logging.debug('Manually forwarded: %s' % message_id)
    if chat_id == -CFG['groupid']:
        logmsg(r)
        irc_send(forward_message_id=message_id)

def forwardmulti(message_ids, chat_id, reply_to_message_id=None):
    failed = False
    message_ids = tuple(message_ids)
    for message_id in message_ids:
        logging.info('forwardMessage: %r' % message_id)
        try:
            r = bot_api('forwardMessage', chat_id=chat_id, from_chat_id=-CFG['groupid'], message_id=message_id)
            logging.debug('Forwarded: %s' % message_id)
        except BotAPIFailed as ex:
            failed = True
            break
    if failed:
        forwardmulti_t(message_ids, chat_id, reply_to_message_id)
        logging.debug('Manually forwarded: %s' % (message_ids,))
    elif chat_id == -CFG['groupid']:
        for message_id in message_ids:
            irc_send(forward_message_id=message_id)

def forwardmulti_t(message_ids, chat_id, reply_to_message_id=None):
    text = []
    for message_id in message_ids:
        m = db_getmsg(message_id)
        if m:
            text.append('[%s] %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(m[4] + CFG['timezone'] * 3600)), db_getufname(m[1]), m[2]))
    sendmsg('\n'.join(text) or 'Found nothing.', chat_id, reply_to_message_id)

def typing(chat_id):
    logging.info('sendChatAction: %r' % chat_id)
    bot_api('sendChatAction', chat_id=chat_id, action='typing')

#def extract_tag(s):
    #words = []
    #tags = []
    #for frag in s.split():
        #if frag[0] == '#':
            ## Should simulate Telegram behavior
            #tags.append(frag[1:])
            #words.extend(jieba.cut(frag[1:]))
        #elif frag[0] == '@':
            #pass
        #else:
            #words.extend(jieba.cut(frag))
    ## counting frequency in a short sentence makes no sense
    #return (words, set(tags))

def daystart(sec=None):
    if not sec:
        sec = time.time()
    return (sec + CFG["timezone"]*3600)//86400 * 86400 - CFG["timezone"]*3600

def uniq(seq): # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

def classify(msg):
    '''
    Classify message type:
    
    - Command: (0)
            All messages that start with a slash ‘/’ (see Commands above)
            Messages that @mention the bot by username
            Replies to the bot's own messages

    - Group message (1)
    - new_chat_participant (2)
    - Ignored message (10)
    - Invalid calling (-1)
    '''
    logging.debug(msg)
    chat = msg['chat']
    text = msg.get('text', '').strip()
    if text:
        if text[0] in "/'" or ('@' + CFG['botname']) in text:
            return 0
        elif 'first_name' in chat:
            return 0
        else:
            reply = msg.get('reply_to_message')
            if reply and reply['from']['id'] == CFG['botid']:
                return 0

    if 'title' in chat:
        # Group chat
        if chat['id'] == -CFG['groupid']:
            if msg['from']['id'] == CFG['botid']:
                return 10
            elif 'new_chat_participant' in msg:
                return 2
            else:
                return 1
        else:
            return 10
    else:
        return -1

def command(text, chatid, replyid, msg):
    try:
        t = text.strip().split(' ')
        if not t:
            return
        if t[0][0] in "/'":
            cmd = t[0][1:].lower().replace('@' + CFG['botname'], '')
            if cmd in COMMANDS:
                if chatid > 0 or chatid == -CFG['groupid'] or cmd in PUBLIC:
                    logging.info('Command: ' + repr(t))
                    COMMANDS[cmd](' '.join(t[1:]).strip(), chatid, replyid, msg)
            elif chatid > 0:
                sendmsg('Invalid command. Send /help for help.', chatid, replyid)
        # 233333
        #elif all(n.isdigit() for n in t):
            #COMMANDS['m'](' '.join(t), chatid, replyid, msg)
        elif chatid != -CFG['groupid']:
            t = ' '.join(t).strip()
            logging.info('Reply: ' + t[:20])
            COMMANDS['reply'](t, chatid, replyid, msg)
    except Exception:
        logging.exception('Excute command failed.')

def async_command(text, chatid, replyid, msg):
    thr = threading.Thread(target=command, args=(text, chatid, replyid, msg))
    thr.run()

def processmsg():
    d = MSG_Q.get()
    logging.debug('Msg arrived: %r' % d)
    uid = d['update_id']
    if 'message' in d:
        msg = d['message']
        if 'text' in msg:
            msg['text'] = msg['text'].replace('\xa0', ' ')
        MSG_CACHE[msg['message_id']] = msg
        cls = classify(msg)
        logging.debug('Classified as: %s', cls)
        if cls == 0:
            async_command(msg['text'], msg['chat']['id'], msg['message_id'], msg)
            if msg['chat']['id'] == -CFG['groupid']:
                logmsg(msg)
        elif cls == 1:
            logmsg(msg)
        elif cls == 2:
            logmsg(msg)
            cmd__welcome(msg['text'], msg['chat']['id'], msg['message_id'], msg)
        elif cls == -1:
            sendmsg('Wrong usage', msg['chat']['id'], msg['message_id'])

def db_adduser(d):
    user = (d['id'], d.get('username'), d.get('first_name'), d.get('last_name'))
    conn.execute('REPLACE INTO users (id, username, first_name, last_name) VALUES (?, ?, ?, ?)', user)
    USER_CACHE[d['id']] = (d.get('username'), d.get('first_name'), d.get('last_name'))
    return user

def db_getuser(uid):
    r = USER_CACHE.get(uid)
    if r is None:
        r = conn.execute('SELECT username, first_name, last_name FROM users WHERE id = ?', (uid,)).fetchone() or (None, None, None)
        USER_CACHE[uid] = r
    return r

def db_getufname(uid):
    name, last = db_getuser(uid)[1:]
    if last:
        name += ' ' + last
    return name

@functools.lru_cache(maxsize=10)
def db_getmsg(mid):
    return conn.execute('SELECT * FROM messages WHERE id = ?', (mid,)).fetchone()

@functools.lru_cache(maxsize=10)
def db_getuidbyname(username):
    uid = conn.execute('SELECT id FROM users WHERE username LIKE ?', (username,)).fetchone()
    if uid:
        return uid[0]


def logmsg(d, iorignore=False):
    src = db_adduser(d['from'])[0]
    text = d.get('text') or d.get('caption', '')
    media = {k:d[k] for k in MEDIA_TYPES.intersection(d.keys())}
    fwd_src = db_adduser(d['forward_from'])[0] if 'forward_from' in d else None
    reply_id = d['reply_to_message']['message_id'] if 'reply_to_message' in d else None
    into = 'INSERT OR IGNORE INTO' if iorignore else 'REPLACE INTO'
    conn.execute(into + ' messages (id, src, text, media, date, fwd_src, fwd_date, reply_id) VALUES (?,?,?,?, ?,?,?,?)',
                 (d['message_id'], src, text, json.dumps(media) if media else None, d['date'], fwd_src, d.get('forward_date'), reply_id))
    logging.info('Logged %s: %s', d['message_id'], d.get('text', '')[:15])

### Commands

def cmd_getmsg(expr, chatid, replyid, msg):
    '''/m <message_id> [...] Get specified message(s) by ID(s).'''
    try:
        mids = tuple(map(int, expr.split()))
    except Exception:
        sendmsg('Syntax error. Usage: ' + cmd_getmsg.__doc__, chatid, replyid)
        return
    forwardmulti(mids, chatid, replyid)

def cmd_context(expr, chatid, replyid, msg):
    '''/context <message_id> [number=2] Show the specified message and its context. max=10'''
    expr = expr.split(' ')
    try:
        if len(expr) > 1:
            mid = max(int(expr[0]), 1)
            limit = max(min(int(expr[1]), 10), 1)
        else:
            mid, limit = int(expr[0]), 2
    except Exception:
        sendmsg('Syntax error. Usage: ' + cmd_context.__doc__, chatid, replyid)
        return
    typing(chatid)
    forwardmulti_t(range(mid - limit, mid + limit + 1), chatid, replyid)

def ellipsisresult(s, find, maxctx=50):
    if find:
        try:
            lnid = s.lower().index(find.lower())
            r = s[max(0, lnid - maxctx):min(len(s), lnid + maxctx)].strip()
            if len(r) < len(s):
                r = '… %s …' % r
            return r
        except ValueError:
            return s
    else:
        return s

re_search_number = re.compile(r'([0-9]+)(,[0-9]+)?')

def cmd_search(expr, chatid, replyid, msg):
    '''/search|/s [@username] [keyword] [number=5|number,offset] Search the group log for recent messages. max(number)=20'''
    username, uid, limit, offset = None, None, 5, 0
    if expr:
        expr = expr.split(' ')
        if len(expr) > 1:
            ma = re_search_number.match(expr[-1])
            if ma:
                expr = expr[:-1]
                limit = max(min(int(ma.group(1)), 20), 1)
                offset = int(ma.group(2)[1:]) if ma.group(2) else 0
        if expr[0][0] == '@':
            username = expr[0][1:]
            keyword = ' '.join(expr[1:])
        else:
            keyword = ' '.join(expr)
    else:
        keyword = ''
    if username:
        uid = db_getuidbyname(username)
    typing(chatid)
    if uid is None:
        keyword = ' '.join(expr)
        sqr = conn.execute("SELECT id, src, text, date FROM messages WHERE text LIKE ? ORDER BY date DESC LIMIT ? OFFSET ?", ('%' + keyword + '%', limit, offset)).fetchall()
    else:
        sqr = conn.execute("SELECT id, src, text, date FROM messages WHERE src = ? AND text LIKE ? ORDER BY date DESC LIMIT ? OFFSET ?", (uid, '%' + keyword + '%', limit, offset)).fetchall()
    result = []
    for mid, fr, text, date in sqr:
        text = ellipsisresult(text, keyword)
        if len(text) > 100:
            text = text[:100] + '…'
        if uid:
            result.append('[%d|%s] %s' % (mid, time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(date + CFG['timezone'] * 3600)), text))
        else:
            result.append('[%d|%s] %s: %s' % (mid, time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(date + CFG['timezone'] * 3600)), db_getufname(fr), text))
    sendmsg('\n'.join(result) or 'Found nothing.', chatid, replyid)

def timestring(minutes):
    h, m = divmod(minutes, 60)
    d, h = divmod(h, 24)
    return (' %d 天' % d if d else '') + (' %d 小时' % h if h else '') + (' %d 分钟' % m if m else '')

def cmd_uinfo(expr, chatid, replyid, msg):
    '''/user|/uinfo [@username] [minutes=1440] Show information about <@username>.'''
    if expr:
        expr = expr.split(' ')
        username = expr[0]
        if not username.startswith('@'):
            uid = msg['from']['id']
            try:
                minutes = min(max(int(expr[0]), 1), 3359733)
            except Exception:
                minutes = 1440
        else:
            uid = db_getuidbyname(username[1:])
            if not uid:
                sendmsg('User not found.', chatid, replyid)
                return
            try:
                minutes = min(max(int(expr[1]), 1), 3359733)
            except Exception:
                minutes = 1440
    else:
        uid = msg['from']['id']
        minutes = 1440
    user = db_getuser(uid)
    uinfoln = []
    if user[0]:
        uinfoln.append('@' + user[0])
    uinfoln.append(db_getufname(uid))
    uinfoln.append('ID: %s' % uid)
    result = [', '.join(uinfoln)]
    r = conn.execute('SELECT src FROM messages WHERE date > ?', (time.time() - minutes * 60,)).fetchall()
    timestr = timestring(minutes)
    if r:
        ctr = collections.Counter(i[0] for i in r)
        if uid in ctr:
            rank = sorted(ctr, key=ctr.__getitem__, reverse=True).index(uid) + 1
            result.append('在最近%s内发了 %s 条消息，占 %.2f%%，位列第 %s。' % (timestr, ctr[uid], ctr[uid]/len(r)*100, rank))
        else:
            result.append('在最近%s内没发消息。' % timestr)
    else:
        result.append('在最近%s内没发消息。' % timestr)
    sendmsg('\n'.join(result), chatid, replyid)

def cmd_stat(expr, chatid, replyid, msg):
    '''/stat [minutes=1440] Show statistics.'''
    try:
        minutes = min(max(int(expr), 1), 3359733)
    except Exception:
        minutes = 1440
    r = conn.execute('SELECT src FROM messages WHERE date > ?', (time.time() - minutes * 60,)).fetchall()
    timestr = timestring(minutes)
    if not r:
        sendmsg('在最近%s内无消息。' % timestr, chatid, replyid)
        return
    ctr = collections.Counter(i[0] for i in r)
    mcomm = ctr.most_common(5)
    count = len(r)
    msg = ['在最近%s内有 %s 条消息，一分钟 %.2f 条。' % (timestr, count, count/minutes)]
    msg.extend('%s: %s 条，%.2f%%' % (db_getufname(k), v, v/count*100) for k, v in mcomm)
    msg.append('其他用户 %s 条，人均 %.2f 条' % (len(r) - sum(v for k, v in mcomm), count / len(ctr)))
    sendmsg('\n'.join(msg), chatid, replyid)

def cmd_digest(expr, chatid, replyid, msg):
    sendmsg('Not implemented.', chatid, replyid)

def cmd_calc(expr, chatid, replyid, msg):
    '''/calc <expr> Calculate <expr>.'''
    # Too many bugs
    if expr:
        r = fx233es.Evaluate(expr)
        if r is not None or fx233es.rformat:
            res = fx233es.PrintResult()
            if len(res) > 200:
                res = res[:200] + '...'
            sendmsg(res or 'Nothing', chatid, replyid)
    else:
        sendmsg('Syntax error. Usage: ' + cmd_calc.__doc__, chatid, replyid)

def cmd_py(expr, chatid, replyid, msg):
    '''/py <expr> Evaluate Python 2 expression <expr>.'''
    if expr:
        if len(expr) > 1000:
            sendmsg('Expression too long.', chatid, replyid)
        else:
            res = geteval(expr)
            if len(res) > 500:
                res = res[:500] + '...'
            sendmsg(res or 'None or error occurred.', chatid, replyid)
    else:
        sendmsg('Syntax error. Usage: ' + cmd_py.__doc__, chatid, replyid)

def cmd_name(expr, chatid, replyid, msg):
    '''/name [pinyin] Get a Chinese name.'''
    surnames, names = namemodel.processinput(expr, 10)
    res = []
    if surnames:
        res.append('姓：' + ', '.join(surnames[:10]))
    if names:
        res.append('名：' + ', '.join(names[:10]))
    sendmsg('\n'.join(res), chatid, replyid)

def cmd_ime(expr, chatid, replyid, msg):
    '''/ime [pinyin] Simple Pinyin IME.'''
    tinput = ''
    if 'reply_to_message' in msg:
        tinput = msg['reply_to_message'].get('text', '')
    tinput = (expr or tinput).strip()
    if len(tinput) > 200:
        tinput = tinput[:200] + '…'
    if not tinput:
        sendmsg('Syntax error. Usage: ' + cmd_ime.__doc__, chatid, replyid)
        return
    res = zhconv.convert(simpleime.pinyininput(tinput), 'zh-hans')
    sendmsg(res, chatid, replyid)

def cmd_quote(expr, chatid, replyid, msg):
    '''/quote Send a today's random message.'''
    typing(chatid)
    sec = daystart()
    msg = conn.execute('SELECT id FROM messages WHERE date >= ? AND date < ? ORDER BY RANDOM() LIMIT 1', (sec, sec + 86400)).fetchone()
    if msg is None:
        msg = conn.execute('SELECT id FROM messages ORDER BY RANDOM() LIMIT 1').fetchone()
    #forwardmulti((msg[0]-1, msg[0], msg[0]+1), chatid, replyid)
    forward(msg[0], chatid, replyid)

def cmd_wyw(expr, chatid, replyid, msg):
    '''/wyw [c|m] <something> Translate something to or from classical Chinese.'''
    if expr[:2].strip() == 'c':
        lang = 'c2m'
        expr = expr[2:]
    elif expr[:2].strip() == 'm':
        lang = 'm2c'
        expr = expr[2:]
    else:
        lang = None
    tinput = ''
    if 'reply_to_message' in msg:
        tinput = msg['reply_to_message'].get('text', '')
    tinput = (expr or tinput).strip()
    if len(tinput) > 800:
        tinput = tinput[:800] + '……'
    if not tinput:
        sendmsg('Syntax error. Usage: ' + cmd_wyw.__doc__, chatid, replyid)
        return
    typing(chatid)
    if lang is None:
        cscore, mscore = zhutil.calctxtstat(tinput)
        if cscore == mscore:
            lang = None
        elif zhutil.checktxttype(cscore, mscore) == 'c':
            lang = 'c2m'
        else:
            lang = 'm2c'
    if lang:
        tres = mosesproxy.translate(tinput, lang, 0, 0, 0)
        sendmsg(tres, chatid, replyid)
    else:
        sendmsg(tinput, chatid, replyid)

def cmd_say(expr, chatid, replyid, msg):
    '''/say Say something interesting.'''
    typing(chatid)
    sendmsg(SAY_Q.get() or 'ERROR_BRAIN_NOT_CONNECTED', chatid, replyid)

def cmd_reply(expr, chatid, replyid, msg):
    '''/reply [question] Reply to the conversation.'''
    typing(chatid)
    text = ''
    if 'reply_to_message' in msg:
        text = msg['reply_to_message'].get('text', '')
    text = (expr.strip() or text or ' '.join(t[0] for t in conn.execute("SELECT text FROM messages ORDER BY date DESC LIMIT 2").fetchall())).replace('\n', ' ')
    r = getsayingbytext(text)
    sendmsg(r or 'ERROR_BRAIN_CONNECT_FAILED', chatid, replyid)

def cmd_echo(expr, chatid, replyid, msg):
    '''/echo Parrot back.'''
    if 'ping' in expr.lower():
        sendmsg('pong', chatid, replyid)
    elif expr:
        sendmsg(expr, chatid, replyid)
    else:
        sendmsg('ping', chatid, replyid)

def cmd__cmd(expr, chatid, replyid, msg):
    global SAY_P
    if chatid < 0:
        return
    if expr == 'reload_model':
        SAY_P.terminate()
        SAY_P = subprocess.Popen(SAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='vendor')
        sendmsg('LM reloaded.', chatid, replyid)
    elif expr == 'commit':
        db.commit()
        sendmsg('DB committed.', chatid, replyid)
    else:
        sendmsg('ping', chatid, replyid)

def cmd_hello(expr, chatid, replyid, msg):
    delta = time.time() - daystart()
    if delta < 6*3600 or delta >= 23*3600:
        sendmsg('还不快点睡觉！', chatid, replyid)
    elif 6*3600 <= delta < 11*3600:
        sendmsg('早上好', chatid, replyid)
    elif 11*3600 <= delta < 13*3600:
        sendmsg('吃饭了没？', chatid, replyid)
    elif 13*3600 <= delta < 18*3600:
        sendmsg('该干嘛干嘛！', chatid, replyid)
    elif 18*3600 <= delta < 23*3600:
        sendmsg('晚上好！', chatid, replyid)

def cmd__welcome(expr, chatid, replyid, msg):
    if chatid > 0:
        return
    usr = msg["new_chat_participant"]
    USER_CACHE[usr["id"]] = (usr.get("username"), usr.get("first_name"), usr.get("last_name"))
    sendmsg('欢迎 %s 加入本群！' % db_getufname(usr["id"]), chatid, replyid)

def cmd_233(expr, chatid, replyid, msg):
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
        txt += '\n' + '(🌝%d/🌚%d)' % (wcount, num - wcount)
    sendmsg(txt, chatid, replyid)

def cmd_start(expr, chatid, replyid, msg):
    if chatid != -CFG['groupid']:
        sendmsg('This is Orz Digger. It can help you search the long and boring chat log of the ##Orz group.\nSend me /help for help.', chatid, replyid)

def cmd_help(expr, chatid, replyid, msg):
    '''/help Show usage.'''
    if chatid == -CFG['groupid']:
        sendmsg('Full help disabled in this group.', chatid, replyid)
    elif chatid > 0:
        sendmsg('\n'.join(uniq(cmd.__doc__ for cmd in COMMANDS.values() if cmd.__doc__)), chatid, replyid)
    else:
        sendmsg('\n'.join(uniq(cmd.__doc__ for cmdname, cmd in COMMANDS.items() if cmd.__doc__ and cmdname in PUBLIC)), chatid, replyid)


# should document usage in docstrings
COMMANDS = collections.OrderedDict((
('m', cmd_getmsg),
('context', cmd_context),
('s', cmd_search),
('search', cmd_search),
('user', cmd_uinfo),
('uinfo', cmd_uinfo),
('digest', cmd_digest),
('stat', cmd_stat),
#('calc', cmd_calc),
('calc', cmd_py),
('py', cmd_py),
('name', cmd_name),
('ime', cmd_ime),
('quote', cmd_quote),
('wyw', cmd_wyw),
('say', cmd_say),
('reply', cmd_reply),
('echo', cmd_echo),
('hello', cmd_hello),
('233', cmd_233),
('start', cmd_start),
('help', cmd_help),
('_cmd', cmd__cmd)
))

PUBLIC = set((
'py',
'name',
'ime',
'wyw',
'say',
'reply',
'echo',
'233',
'start',
'help'
))

srandom = random.SystemRandom()

OFFSET = conn.execute('SELECT val FROM config WHERE id = 0').fetchone()
OFFSET = OFFSET[0] if OFFSET else 0
IRCOFFSET = conn.execute('SELECT val FROM config WHERE id = 1').fetchone()
IRCOFFSET = IRCOFFSET[0] if IRCOFFSET else -1000000
USER_CACHE = LRUCache(20)
MSG_CACHE = LRUCache(10)
CFG = json.load(open('config.json'))
URL = 'https://api.telegram.org/bot%s/' % CFG['token']

#importdb('telegram-history.db')
#importupdates(OFFSET, 2000)

MSG_Q = queue.Queue()
SAY_Q = queue.Queue(maxsize=50)
SAY_LCK = threading.Lock()
EVIL_LCK = threading.Lock()

SAY_CMD = ('python3', 'say.py', 'chat.binlm', 'chatdict.txt', 'context.pkl')
SAY_P = subprocess.Popen(SAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='vendor')

EVIL_CMD = ('python', 'seccomp.py')
EVIL_P = subprocess.Popen(EVIL_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='vendor')

pollthr = threading.Thread(target=getupdates)
pollthr.daemon = True
pollthr.start()

saythr = threading.Thread(target=getsaying)
saythr.daemon = True
saythr.start()

ircconn = None
if 'ircserver' in CFG:
    checkircconn()
    ircthr = threading.Thread(target=getircupd)
    ircthr.daemon = True
    ircthr.start()

# fx233es = fparser.Parser(numtype='decimal')

namemodel = chinesename.NameModel('vendor/namemodel.m')
simpleime.loaddict('vendor/pyindex.dawg', 'vendor/essay.dawg')

logging.info('Satellite launched.')

try:
    while 1:
        try:
            processmsg()
        except Exception as ex:
            logging.exception('Process a message failed.')
            continue
finally:
    conn.execute('REPLACE INTO config (id, val) VALUES (0, ?)', (OFFSET,))
    conn.execute('REPLACE INTO config (id, val) VALUES (1, ?)', (IRCOFFSET,))
    db.commit()
    SAY_P.terminate()
    logging.info('Shut down cleanly.')
