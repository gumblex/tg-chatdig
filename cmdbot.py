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
import datetime
import threading
import functools
import subprocess
import collections

import requests

__version__ = '1.0'

MEDIA_TYPES = frozenset(('audio', 'document', 'photo', 'sticker', 'video', 'contact', 'location', 'new_chat_participant', 'left_chat_participant', 'new_chat_title', 'new_chat_photo', 'delete_chat_photo', 'group_chat_created'))

logging.basicConfig(stream=sys.stdout, format='# %(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

HSession = requests.Session()
USERAGENT = 'TgCmdBot/%s %s' % (__version__, HSession.headers["User-Agent"])
HSession.headers["User-Agent"] = USERAGENT

db = sqlite3.connect('botstate.db')
conn = db.cursor()
conn.execute('''CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY,
username TEXT,
first_name TEXT,
last_name TEXT
)''')
conn.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, val INTEGER)')


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

def geteval(text=''):
    global EVIL_P
    with EVIL_LCK:
        if EVIL_P.returncode is not None:
            EVIL_P = subprocess.Popen(EVIL_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            result, errs = EVIL_P.communicate(text.strip().encode('utf-8'), timeout=10)
        except Exception: # TimeoutExpired
            EVIL_P.kill()
            result, errs = EVIL_P.communicate()
        result = result.strip().decode('utf-8', errors='replace')
    return result

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
    while 1:
        try:
            req = HSession.get(URL + method, params=params)
            ret = json.loads(req.text)
            break
        except Exception as ex:
            change_session()
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
    logging.info('sendMessage: %s' % text[:20])
    m = bot_api('sendMessage', chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id)
    logmsg(m)

def typing(chat_id):
    logging.info('sendChatAction: %r' % chat_id)
    bot_api('sendChatAction', chat_id=chat_id, action='typing')

def daystart(sec=None):
    if not sec:
        sec = time.time()
    return (sec + CFG["timezone"]*3600)//86400 * 86400 - CFG["timezone"]*3600

def classify(msg):
    '''
    Classify message type:
    
    - Command: (0)
            All messages that start with a slash ‘/’ (see Commands above)
            Messages that @mention the bot by username
            Replies to the bot's own messages

    - Group message (1)
    - Ignored message (2)
    - Invalid calling (-1)
    '''
    logging.debug(msg)
    chat = msg['chat']
    text = msg.get('text')
    if text:
        if text.startswith('/') or ('@' + CFG['botname']) in text:
            return 0
        elif 'first_name' in chat:
            return 0
        else:
            reply = msg.get('reply_to_message')
            if reply and reply['from']['id'] == CFG['botid']:
                return 0

    if 'title' in chat:
        # Group chat
        if msg['from']['id'] == CFG['botid']:
            return 2
        return 1
    else:
        return -1

def command(text, chatid, replyid):
    try:
        t = text.strip().split(' ')
        if not t:
            return
        if t[0][0] == '/':
            cmd = t[0][1:].lower().replace('@' + CFG['botname'], '')
            if cmd in COMMANDS:
                logging.info('Command: ' + repr(t))
                COMMANDS[cmd](' '.join(t[1:]), chatid, replyid)
            elif chatid > 0:
                sendmsg('Invalid command. Send /help for help.', chatid, replyid)
        elif chatid > 0:
            t = ' '.join(t)
            logging.info('Reply: ' + t[:20])
            COMMANDS['reply'](t, chatid, replyid)
    except Exception:
        logging.exception('Excute command failed.')

def async_command(text, chatid, replyid):
    thr = threading.Thread(target=command, args=(text, chatid, replyid))
    thr.run()

def processmsg():
    d = MSG_Q.get()
    logging.debug('Msg arrived: %r' % d)
    uid = d['update_id']
    if 'message' in d:
        msg = d['message']
        MSG_CACHE[msg['message_id']] = msg
        cls = classify(msg)
        logging.debug('Classified as: %s', cls)
        if cls == 0:
            async_command(msg['text'], msg['chat']['id'], msg['message_id'])
            logmsg(msg)
        elif cls == 1:
            logmsg(msg)
            if 'new_chat_participant' in msg:
                sendmsg('欢迎 %s 加入本群！' % dc_getufname(msg["new_chat_participant"]), chatid, replyid)
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

def dc_getufname(userdict):
    name = userdict['first_name']
    if 'last_name' in userdict:
        name += ' ' + userdict['last_name']
    return name

def logmsg(d):
    src = db_adduser(d['from'])[0]
    fwd_src = db_adduser(d['forward_from'])[0] if 'forward_from' in d else None
    logging.info('Logged users in %s', d['message_id'])

### Commands

def ellipsisresult(s, find, maxctx=50):
    lnid = s.lower().index(find.lower())
    r = s[max(0, lnid - maxctx):min(len(s), lnid + maxctx)].strip()
    if len(r) < len(s):
        r = '… %s …' % r
    return r

re_search_number = re.compile(r'([0-9]+)(,[0-9]+)?')

def cmd_whoami(expr, chatid, replyid):
    '''/whoami Show information about you.'''
    origmsg = MSG_CACHE.get(replyid, {})
    if origmsg:
        result = []
        d = origmsg['from']
        if 'username' in d:
            result.append('@' + d['username'])
        name = d['first_name']
        if 'last_name' in d:
            name += ' ' + d['last_name']
        result.append(name)
        result.append('id: %s' % d['id'])
        sendmsg(', '.join(result), chatid, replyid)

def cmd_py(expr, chatid, replyid):
    '''/py <expr> Evaluate Python 2 expression <expr>.'''
    if expr:
        if len(expr) > 500:
            sendmsg('Expression too long.', chatid, replyid)
        else:
            res = geteval(expr)
            if len(res) > 300:
                res = res[:300] + '...'
            sendmsg(res or 'None or error occurred.', chatid, replyid)
    else:
        sendmsg('Syntax error. Usage: ' + cmd_py.__doc__, chatid, replyid)

def cmd_reply(expr, chatid, replyid):
    # '''/reply [question] Reply to the conversation.'''
    sendmsg('Not implemented.', chatid, replyid)
    return
    ################################################
    typing(chatid)
    origmsg = MSG_CACHE.get(replyid, {})
    text = ''
    if 'reply_to_message' in origmsg:
        text = origmsg['reply_to_message'].get('text', '')
    text = (expr.strip() or text).replace('\n', ' ')
    r = getsayingbytext(text)
    sendmsg(r or 'ERROR_BRAIN_CONNECT_FAILED', chatid, replyid)

def cmd_echo(expr, chatid, replyid):
    '''/echo Parrot back.'''
    if 'ping' in expr.lower():
        sendmsg('pong', chatid, replyid)
    elif expr:
        sendmsg(expr, chatid, replyid)
    else:
        sendmsg('ping', chatid, replyid)

def cmd_233(expr, chatid, replyid):
    try:
        num = max(min(int(expr), 100), 1)
    except Exception:
        num = 1
    w = math.ceil(num ** .5)
    h, rem = divmod(num, w)
    txt = '\n'.join(''.join(srandom.choice('🌝🌚') for i in range(w)) for j in range(h))
    if rem:
        txt += '\n' + ''.join(srandom.choice('🌝🌚') for i in range(rem))
    sendmsg(txt, chatid, replyid)

def cmd_start(expr, chatid, replyid):
    if chatid > 0:
        sendmsg('Yet another Telegram bot.\nSend me /help for help.', chatid, replyid)

def cmd_help(expr, chatid, replyid):
    '''/help Show usage.'''
    sendmsg('\n'.join(cmd.__doc__ for cmd in COMMANDS.values() if cmd.__doc__), chatid, replyid)


# should document usage in docstrings
COMMANDS = collections.OrderedDict((
('whoami', cmd_whoami),
('py', cmd_py),
('echo', cmd_echo),
('233', cmd_233),
('start', cmd_start),
('help', cmd_help)
))

srandom = random.SystemRandom()

OFFSET = conn.execute('SELECT val FROM config WHERE id = 0').fetchone()
OFFSET = OFFSET[0] if OFFSET else 0
USER_CACHE = LRUCache(20)
MSG_CACHE = LRUCache(20)
CFG = json.load(open('cmdbot.json'))
URL = 'https://api.telegram.org/bot%s/' % CFG['token']

MSG_Q = queue.Queue()
EVIL_LCK = threading.Lock()

EVIL_CMD = ('python', 'vendor/seccomp.py')
EVIL_P = subprocess.Popen(EVIL_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

pollthr = threading.Thread(target=getupdates)
pollthr.daemon = True
pollthr.start()

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
    db.commit()
    logging.info('Shut down cleanly.')
