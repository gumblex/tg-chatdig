#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import math
import time
import json
import queue
import signal
import socket
import random
import logging
import sqlite3
import threading
import functools
import subprocess
import collections
import unicodedata
import concurrent.futures

import requests
from vendor import libirc

__version__ = '1.4'

MEDIA_TYPES = frozenset(('audio', 'document', 'photo', 'sticker', 'video', 'voice', 'contact', 'location', 'new_chat_participant', 'left_chat_participant', 'new_chat_title', 'new_chat_photo', 'delete_chat_photo', 'group_chat_created'))
EXT_MEDIA_TYPES = frozenset(('audio', 'document', 'photo', 'sticker', 'video', 'voice', 'contact', 'location', 'new_chat_participant', 'left_chat_participant', 'new_chat_title', 'new_chat_photo', 'delete_chat_photo', 'group_chat_created', '_ircuser'))

loglevel = logging.DEBUG if sys.argv[-1] == '-d' else logging.INFO

logging.basicConfig(stream=sys.stdout, format='# %(asctime)s [%(levelname)s] %(message)s', level=loglevel)

socket.setdefaulttimeout(60)

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

re_ircaction = re.compile('^\x01ACTION (.*)\x01$')
re_ircforward = re.compile(r'^\[([^]]+)\] (.*)$|^\*\* ([^ ]+) (.*) \*\*$')

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

def async_func(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        def func_noerr(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception:
                logging.exception('Async function failed.')
        executor.submit(func_noerr, *args, **kwargs)
    return wrapped

def _raise_ex(ex):
    raise ex

### Polling

def getupdates():
    global OFFSET, MSG_Q
    while 1:
        try:
            updates = bot_api('getUpdates', offset=OFFSET, timeout=10)
        except Exception as ex:
            logging.exception('Get updates failed.')
            continue
        if updates:
            logging.debug('Messages coming.')
            OFFSET = updates[-1]["update_id"] + 1
            for upd in updates:
                MSG_Q.put(upd)
        time.sleep(.2)

def checkappproc():
    global APP_P
    if APP_P.poll() is not None:
        APP_P = subprocess.Popen(APP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def runapptask(cmd, args, sendargs):
    '''`sendargs` should be (chatid, replyid)'''
    global APP_P, APP_LCK, APP_TASK
    with APP_LCK:
        # Prevent float problems
        tid = str(time.time())
        text = json.dumps({"cmd": cmd, "args": args, "id": tid})
        APP_TASK[tid] = sendargs
        try:
            APP_P.stdin.write(text.strip().encode('utf-8') + b'\n')
            APP_P.stdin.flush()
        except BrokenPipeError:
            checkappproc()
            APP_P.stdin.write(text.strip().encode('utf-8') + b'\n')
            APP_P.stdin.flush()
        logging.debug('Wrote to APP_P: ' + text)

def getappresult():
    global APP_P, APP_TASK
    while 1:
        try:
            result = APP_P.stdout.readline().strip().decode('utf-8')
        except BrokenPipeError:
            checkappproc()
            result = APP_P.stdout.readline().strip().decode('utf-8')
        logging.debug('Got from APP_P: ' + result)
        if result:
            obj = json.loads(result)
            if obj['exc']:
                logging.error('Remote app server error.\n' + obj['exc'])
            sargs = APP_TASK.get(obj['id'])
            if sargs:
                sendmsg(obj['ret'] or 'Empty.', sargs[0], sargs[1])
                del APP_TASK[obj['id']]
            else:
                logging.error('Task ID %s not found.' % obj['id'])

def checkircconn():
    global ircconn
    if not ircconn or not ircconn.sock:
        ircconn = libirc.IRCConnection()
        ircconn.connect((CFG['ircserver'], CFG['ircport']), use_ssl=CFG['ircssl'])
        if CFG.get('ircpass'):
            ircconn.setpass(CFG['ircpass'])
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
            if line["dest"] != CFG['ircnick'] and not re.match(CFG['ircignore'], line["nick"]):
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

def ircconn_say(dest, msg, sendnow=True):
    MIN_INT = 0.2
    if not ircconn:
        return
    curtime = time.time()
    delta = curtime - ircconn_say.lasttime
    if delta < MIN_INT:
        time.sleep(MIN_INT - delta)
    ircconn.say(dest, msg, sendnow)
    ircconn_say.lasttime = time.time()
ircconn_say.lasttime = 0

def irc_send(text='', reply_to_message_id=None, forward_message_id=None):
    if ircconn:
        checkircconn()
        if reply_to_message_id:
            m = MSG_CACHE.get(reply_to_message_id, {})
            logging.debug('Got reply message: ' + str(m))
            if '_ircuser' in m:
                text = "%s: %s" % (m['_ircuser'], text)
            elif 'from' in m:
                src = smartname(m['from'])
                if m['from']['id'] in (CFG['botid'], CFG['ircbotid']):
                    rnmatch = re_ircforward.match(m.get('text', ''))
                    if rnmatch:
                        src = rnmatch.group(1) or src
                text = "%s: %s" % (src, text)
        elif forward_message_id:
            # not async, so no sqlite3.ProgrammingError in db_*
            m = db_getmsg(forward_message_id)
            if m:
                text = "Fwd %s: %s" % (smartname(m[1], True), m[2])
        lines = text.splitlines()
        if len(lines) < 3:
            text = ' '.join(lines)
        else:
            text = lines[0] + ' [...] ' + lines[-1]
        ircconn_say(CFG['ircchannel'], text)

@async_func
def irc_forward(msg):
    if not ircconn:
        return
    try:
        if msg['from']['id'] == CFG['ircbotid']:
            return
        checkircconn()
        text = msg.get('text', '')
        mkeys = tuple(msg.keys() & MEDIA_TYPES)
        if mkeys:
            if text:
                text += ' ' + servemedia(msg)
            else:
                text = servemedia(msg)
        if text and not text.startswith('@@@'):
            if 'forward_from' in msg:
                fwdname = ''
                if msg['forward_from']['id'] in (CFG['botid'], CFG['ircbotid']):
                    rnmatch = re_ircforward.match(msg.get('text', ''))
                    if rnmatch:
                        fwdname = rnmatch.group(1) or rnmatch.group(3)
                        text = rnmatch.group(2) or rnmatch.group(4)
                fwdname = fwdname or smartname(msg['forward_from'])
                text = "Fwd %s: %s" % (fwdname, text)
            elif 'reply_to_message' in msg:
                replname = ''
                replyu = msg['reply_to_message']['from']
                if replyu['id'] in (CFG['botid'], CFG['ircbotid']):
                    rnmatch = re_ircforward.match(msg['reply_to_message'].get('text', ''))
                    if rnmatch:
                        replname = rnmatch.group(1) or rnmatch.group(3)
                replname = replname or smartname(replyu)
                text = "%s: %s" % (replname, text)
            # ignore blank lines
            text = list(filter(lambda s: s.strip(), text.splitlines()))
            if len(text) > 3:
                text = text[:3]
                text[-1] += ' [...]'
            for ln in text[:3]:
                ircconn_say(CFG['ircchannel'], '[%s] %s' % (smartname(msg['from']), ln))
    except Exception:
        logging.exception('Forward a message to IRC failed.')

### DB import

def mediaformatconv(media=None, action=None):
    type_map = {
    # media
    'photo': 'photo',
    'document': 'document',
    'unsupported': 'document',
    'geo': 'location',
    'venue': 'location',
    'contact': 'contact',
    # action
    'chat_add_user': 'new_chat_participant',
    'chat_add_user_link': 'new_chat_participant',
    'chat_del_user': 'left_chat_participant',
    'chat_rename': 'new_chat_title',
    'chat_change_photo': 'new_chat_photo',
    'chat_delete_photo': 'delete_chat_photo',
    'chat_created': 'group_chat_created'
    }
    d = {}
    caption = None
    if media:
        media = json.loads(media)
    if action:
        action = json.loads(action)
    if media and 'type' in media:
        media = media.copy()
        if media['type'] == 'photo':
            caption = media['caption']
            d['photo'] = []
        elif media['type'] in ('document', 'unsupported'):
            d['document'] = {}
        elif 'longitude' in media:
            # 'type' may be the name of the place
            d['location'] = {
                'longitude': media['longitude'],
                'latitude': media['latitude']
            }
        elif media['type'] == 'contact':
            del media['type']
            media['phone_number'] = media.pop('phone')
            d['contact'] = media
        # ignore other undefined types to Bot API
    if action and 'type' in action:
        newname = type_map.get(action['type'])
        if newname.endswith('chat_participant'):
            d[newname] = {
                'id': action['user']['id'],
                'first_name': action['user'].get('first_name', ''),
                'last_name': action['user'].get('last_name', ''),
                'username': action['user'].get('username', '')
            }
        elif newname == 'new_chat_title':
            d[newname] = action['title']
        elif newname == 'new_chat_photo':
            d[newname] = []
        elif newname in ('delete_chat_photo', 'group_chat_created'):
            d[newname] = True
        # ignore other undefined types to Bot API
    return json.dumps(d) if d else None, caption

def importdb(filename):
    logging.info('Import DB...')
    if not os.path.isfile(filename):
        logging.warning('DB not found.')
        return
    db_s = sqlite3.connect(filename)
    conn_s = db_s.cursor()
    for vals in conn_s.execute('SELECT id, src, text, media, date, fwd_src, fwd_date, reply_id, action FROM messages WHERE dest = ?', (CFG['groupid'],)):
        vals = list(vals)
        vals[0] = -250000 + vals[0]
        vals[3], caption = mediaformatconv(vals[3], vals.pop())
        vals[2] = vals[2] or caption
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
            if 'message' in d:
                msg = d['message']
                cls = classify(msg)
                if cls == 0 and msg['chat']['id'] == -CFG['groupid']:
                    logmsg(msg, True)
                elif cls == 1:
                    logmsg(msg, True)
        time.sleep(.1)
        updates = bot_api('getUpdates', offset=off, limit=100)

def importfixservice(filename):
    logging.info('Updating DB...')
    if not os.path.isfile(filename):
        logging.warning('DB not found.')
        return
    db_s = sqlite3.connect(filename)
    conn_s = db_s.cursor()
    for mid, text, media, action in conn_s.execute('SELECT id, text, media, action FROM messages WHERE dest = ?', (CFG['groupid'],)):
        mid -= 250000
        media, caption = mediaformatconv(media, action)
        text = text or caption
        conn.execute('UPDATE messages SET text=?, media=? WHERE id=?', (text, media, mid))
    db.commit()
    logging.info('Fix DB media column done.')

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
    for att in range(3):
        try:
            req = HSession.get(URL + method, params=params, timeout=45)
            retjson = req.content
            ret = json.loads(retjson.decode('utf-8'))
            break
        except Exception as ex:
            if att < 1:
                time.sleep((att+1) * 2)
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

def sync_sendmsg(text, chat_id, reply_to_message_id=None):
    global LOG_Q
    text = text.strip()
    if not text:
        logging.warning('Empty message ignored: %s, %s' % (chat_id, reply_to_message_id))
        return
    logging.info('sendMessage(%s): %s' % (len(text), text[:20]))
    if len(text) > 2000:
        text = text[:1999] + '…'
    reply_id = reply_to_message_id
    if reply_to_message_id and reply_to_message_id < 0:
        reply_id = None
    m = bot_api('sendMessage', chat_id=chat_id, text=text, reply_to_message_id=reply_id)
    if chat_id == -CFG['groupid']:
        MSG_CACHE[m['message_id']] = m
        # IRC messages
        if reply_to_message_id is not None:
            LOG_Q.put(m)
            irc_send(text, reply_to_message_id)
    return m

sendmsg = async_func(sync_sendmsg)

#@async_func
def forward(message_id, chat_id, reply_to_message_id=None):
    global LOG_Q
    logging.info('forwardMessage: %r' % message_id)
    try:
        if message_id < 0:
            raise ValueError('Invalid message id')
        r = bot_api('forwardMessage', chat_id=chat_id, from_chat_id=-CFG['groupid'], message_id=message_id)
        logging.debug('Forwarded: %s' % message_id)
    except (ValueError, BotAPIFailed) as ex:
        m = db_getmsg(message_id)
        if m:
            r = sendmsg('[%s] %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(m[4] + CFG['timezone'] * 3600)), db_getufname(m[1]), m[2]), chat_id, reply_to_message_id)
            logging.debug('Manually forwarded: %s' % message_id)
    if chat_id == -CFG['groupid']:
        LOG_Q.put(r)
        irc_send(forward_message_id=message_id)

#@async_func
def forwardmulti(message_ids, chat_id, reply_to_message_id=None):
    failed = False
    message_ids = tuple(message_ids)
    for message_id in message_ids:
        logging.info('forwardMessage: %r' % message_id)
        try:
            if message_id < 0:
                raise ValueError('Invalid message id')
            r = bot_api('forwardMessage', chat_id=chat_id, from_chat_id=-CFG['groupid'], message_id=message_id)
            logging.debug('Forwarded: %s' % message_id)
            if chat_id == -CFG['groupid']:
                LOG_Q.put(r)
        except (ValueError, BotAPIFailed) as ex:
            failed = True
            break
    if failed:
        forwardmulti_t(message_ids, chat_id, reply_to_message_id)
        logging.debug('Manually forwarded: %s' % (message_ids,))
    elif chat_id == -CFG['groupid']:
        for message_id in message_ids:
            irc_send(forward_message_id=message_id)

#@async_func
def forwardmulti_t(message_ids, chat_id, reply_to_message_id=None):
    text = []
    for message_id in message_ids:
        m = db_getmsg(message_id)
        if m:
            text.append('[%s] %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(m[4] + CFG['timezone'] * 3600)), db_getufname(m[1]), m[2]))
    sendmsg('\n'.join(text) or 'Message(s) not found.', chat_id, reply_to_message_id)

@async_func
def typing(chat_id):
    logging.info('sendChatAction: %r' % chat_id)
    bot_api('sendChatAction', chat_id=chat_id, action='typing')

def getfile(file_id):
    logging.info('getFile: %r' % file_id)
    return bot_api('getFile', file_id=file_id)

def retrieve(url, filename, raisestatus=True):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    if raisestatus:
        r.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
        f.flush()
    return r.status_code

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
    - IRC message (2)
    - new_chat_participant (3)
    - Ignored message (10)
    - Invalid calling (-1)
    '''
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

    # If not enabled, there won't be this kind of msg
    ircu = msg.get('_ircuser')
    if ircu and ircu != CFG['ircnick']:
        return 2

    if 'title' in chat:
        # Group chat
        if 'new_chat_participant' in msg:
            return 3
        if chat['id'] == -CFG['groupid']:
            if msg['from']['id'] == CFG['botid']:
                return 10
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
                    expr = ' '.join(t[1:]).strip()
                    logging.info('Command: /%s %s' % (cmd, expr[:20]))
                    COMMANDS[cmd](expr, chatid, replyid, msg)
                elif chatid < 0 and chatid != -CFG['groupid'] and cmd not in PUBLIC:
                    sendmsg('This command is not available for this group. Send /help for available commands.', chatid, replyid)
            elif chatid > 0:
                sendmsg('Invalid command. Send /help for help.', chatid, replyid)
        # 233333
        #elif all(n.isdigit() for n in t):
            #COMMANDS['m'](' '.join(t), chatid, replyid, msg)
        elif chatid > 0:
            t = ' '.join(t).strip()
            logging.info('Reply: ' + t[:20])
            COMMANDS['reply'](t, chatid, replyid, msg)
    except Exception:
        logging.exception('Excute command failed.')

def processmsg():
    d = MSG_Q.get()
    logging.debug('Msg arrived: %r' % d)
    if 'message' in d:
        msg = d['message']
        if 'text' in msg:
            msg['text'] = msg['text'].replace('\xa0', ' ')
        elif 'caption' in msg:
            msg['text'] = msg['caption'].replace('\xa0', ' ')
        MSG_CACHE[msg['message_id']] = msg
        cls = classify(msg)
        logging.debug('Classified as: %s', cls)
        if msg['chat']['id'] == -CFG['groupid'] and CFG.get('t2i'):
            irc_forward(msg)
        if cls == 0:
            if msg['chat']['id'] == -CFG['groupid']:
                logmsg(msg)
            rid = msg['message_id']
            if CFG.get('i2t') and '_ircuser' in msg:
                rid = sync_sendmsg('[%s] %s' % (msg['_ircuser'], msg['text']), msg['chat']['id'])['message_id']
            command(msg['text'], msg['chat']['id'], rid, msg)
        elif cls == 1:
            logmsg(msg)
        elif cls == 2:
            logmsg(msg)
            if CFG.get('i2t'):
                act = re_ircaction.match(msg['text'])
                if act:
                    sendmsg('** %s %s **' % (msg['_ircuser'], act.group(1)), msg['chat']['id'])
                else:
                    sendmsg('[%s] %s' % (msg['_ircuser'], msg['text']), msg['chat']['id'])
        elif cls == 3:
            logmsg(msg)
            cmd__welcome('', msg['chat']['id'], msg['message_id'], msg)
        elif cls == -1:
            sendmsg('Wrong usage', msg['chat']['id'], msg['message_id'])
        if cls in (1, 2) and CFG.get('autoclose') and 'forward_from' not in msg:
            autoclose(msg)
        try:
            logmsg(LOG_Q.get_nowait())
        except queue.Empty:
            pass

def cachemedia(msg):
    '''
    Download specified media if not exist.
    '''
    mt = msg.keys() & frozenset(('audio', 'document', 'sticker', 'video', 'voice'))
    file_ext = ''
    if mt:
        mt = mt.pop()
        file_id = msg[mt]['file_id']
        file_size = msg[mt].get('file_size')
        if mt == 'sticker':
            file_ext = '.webp'
    elif 'photo' in msg:
        photo = max(msg['photo'], key=lambda x: x['width'])
        file_id = photo['file_id']
        file_size = photo.get('file_size')
        file_ext = '.jpg'
    fp = getfile(file_id)
    file_size = fp.get('file_size') or file_size
    file_path = fp.get('file_path')
    if not file_path:
        raise BotAPIFailed("can't get file_path for " + file_id)
    file_ext = os.path.splitext(file_path)[1] or file_ext
    cachename = file_id + file_ext
    fpath = os.path.join(CFG['cachepath'], cachename)
    try:
        if os.path.isfile(fpath) and os.path.getsize(fpath) == file_size:
            return (cachename, 304)
    except Exception:
        pass
    return (cachename, retrieve(URL_FILE + file_path, fpath))

def timestring_a(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)

def servemedia(msg):
    '''
    Reply type and link of media. This only generates links for photos.
    '''
    keys = tuple(msg.keys() & MEDIA_TYPES)
    if not keys:
        return ''
    ret = '<%s>' % keys[0]
    if 'photo' in msg:
        servemode = CFG.get('servemedia')
        if servemode:
            fname, code = cachemedia(msg)
            if servemode == 'self':
                ret += ' %s%s' % (CFG['serveurl'], fname)
            elif servemode == 'vim-cn':
                r = requests.post('http://img.vim-cn.com/', files={'name': open(os.path.join(CFG['cachepath'], fname), 'rb')})
                ret += ' ' + r.text
    elif 'sticker' in msg:
        if CFG.get('servemedia') == 'self':
            fname, code = cachemedia(msg)
            ret += ' %s%s' % (CFG['serveurl'], fname)
    elif 'document' in msg:
        ret += ' %s' % (msg['document'].get('file_name', ''))
        if CFG.get('servemedia') == 'self' and msg['document'].get('file_size', 0) <= CFG.get('servemaxsize', 1048576):
            fname, code = cachemedia(msg)
            ret += ' %s%s' % (CFG['serveurl'], fname)
    elif 'video' in msg:
        ret += ' ' + timestring_a(msg['video'].get('duration', 0))
        if CFG.get('servemedia') == 'self' and msg['video'].get('file_size', 0) <= CFG.get('servemaxsize', 1048576):
            fname, code = cachemedia(msg)
            ret += ' %s%s' % (CFG['serveurl'], fname)
    elif 'voice' in msg:
        ret += ' ' + timestring_a(msg['voice'].get('duration', 0))
        if CFG.get('servemedia') == 'self' and msg['voice'].get('file_size', 0) <= CFG.get('servemaxsize', 1048576):
            fname, code = cachemedia(msg)
            ret += ' %s%s' % (CFG['serveurl'], fname)
    elif 'new_chat_title' in msg:
        ret += ' ' + msg['new_chat_title']
    return ret

def autoclose(msg):
    openbrckt = ('([{（［｛⦅〚⦃“‘‹«「〈《【〔⦗『〖〘｢⟦⟨⟪⟮⟬⌈⌊⦇⦉❛❝❨❪❴❬❮❰❲'
                 '⏜⎴⏞〝︵⏠﹁﹃︹︻︗︿︽﹇︷〈⦑⧼﹙﹛﹝⁽₍⦋⦍⦏⁅⸢⸤⟅⦓⦕⸦⸨｟⧘⧚⸜⸌⸂⸄⸉᚛༺༼')
    clozbrckt = (')]}）］｝⦆〛⦄”’›»」〉》】〕⦘』〗〙｣⟧⟩⟫⟯⟭⌉⌋⦈⦊❜❞❩❫❵❭❯❱❳'
                 '⏝⎵⏟〞︶⏡﹂﹄︺︼︘﹀︾﹈︸〉⦒⧽﹚﹜﹞⁾₎⦌⦎⦐⁆⸣⸥⟆⦔⦖⸧⸩｠⧙⧛⸝⸍⸃⸅⸊᚜༻༽')
    stack = []
    for ch in msg.get('text', ''):
        index = openbrckt.find(ch)
        if index >= 0:
            stack.append(index)
            continue
        index = clozbrckt.find(ch)
        if index >= 0:
            if stack and stack[-1] == index:
                stack.pop()
    closed = ''.join(reversed(tuple(map(clozbrckt.__getitem__, stack))))
    if closed:
        if len(closed) > 20:
            closed = closed[:20] + '…'
        sendmsg(closed, msg['chat']['id'], msg['message_id'])

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

def dc_getufname(user, maxlen=100):
    USER_CACHE[user['id']] = (user.get('username'), user.get('first_name'), user.get('last_name'))
    name = user['first_name']
    if 'last_name' in user:
        name += ' ' + user['last_name']
    if len(name) > maxlen:
        name = name[:maxlen] + '…'
    return name

def smartname(user, db=False, limit=20):
    if db:
        first, last = db_getuser(user)[1:]
    else:
        USER_CACHE[user['id']] = (user.get('username'), user.get('first_name'), user.get('last_name'))
        first, last = user.get('first_name', ''), user.get('last_name', '')
    if not first:
        return '<%s>' % 'Unknown'[:limit-2]
    pn = first
    if last:
        pn += ' ' + last
    if len(pn) > limit:
        if len(first) > limit:
            return first.split(None, 1)[0][:limit]
        else:
            return first[:limit]
    else:
        return pn

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
    media = {k:d[k] for k in EXT_MEDIA_TYPES.intersection(d.keys())}
    fwd_src = db_adduser(d['forward_from'])[0] if 'forward_from' in d else None
    reply_id = d['reply_to_message']['message_id'] if 'reply_to_message' in d else None
    into = 'INSERT OR IGNORE INTO' if iorignore else 'REPLACE INTO'
    conn.execute(into + ' messages (id, src, text, media, date, fwd_src, fwd_date, reply_id) VALUES (?,?,?,?, ?,?,?,?)',
                 (d['message_id'], src, text, json.dumps(media) if media else None, d['date'], fwd_src, d.get('forward_date'), reply_id))
    logging.info('Logged %s: %s', d['message_id'], d.get('text', '')[:15])
    db.commit()

### Commands

def cmd_getmsg(expr, chatid, replyid, msg):
    '''/m <message_id> [...] Get specified message(s) by ID(s).'''
    try:
        if not expr:
            if 'reply_to_message' in msg:
                sendmsg('Message ID: %d' % msg['reply_to_message']['message_id'], chatid, replyid)
            else:
                raise ValueError
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

def cmd_quote(expr, chatid, replyid, msg):
    '''/quote Send a today's random message.'''
    typing(chatid)
    sec = daystart()
    msg = conn.execute('SELECT id FROM messages WHERE date >= ? AND date < ? ORDER BY RANDOM() LIMIT 1', (sec, sec + 86400)).fetchone()
    if msg is None:
        msg = conn.execute('SELECT id FROM messages ORDER BY RANDOM() LIMIT 1').fetchone()
    #forwardmulti((msg[0]-1, msg[0], msg[0]+1), chatid, replyid)
    forward(msg[0], chatid, replyid)

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

def cmd_mention(expr, chatid, replyid, msg):
    '''/mention Show last mention of you.'''
    if msg['chat']['id'] != -CFG['groupid']:
        sendmsg("This command can't be used in this chat.", chatid, replyid)
        return
    uid = msg['from']['id']
    user = db_getuser(uid)
    if user[0]:
        res = conn.execute("SELECT * FROM messages WHERE (text LIKE ? OR reply_id IN (SELECT id FROM messages WHERE src = ?)) AND src != ? ORDER BY date DESC LIMIT 1", ('%@' + user[0] + '%', uid, CFG['botid'])).fetchone()
        userat = '@' + user[0] + ' '
    else:
        res = conn.execute("SELECT * FROM messages WHERE reply_id IN (SELECT id FROM messages WHERE src = ?) AND src != ? ORDER BY date DESC LIMIT 1", (uid, CFG['botid'])).fetchone()
        userat = ''
    if res:
        reid = res[0]
        if reid > 0:
            sendmsg(userat + 'You were mentioned in this message.', chatid, reid)
        else:
            forward(reid, chatid, replyid)
    else:
        sendmsg('No mention found.', chatid, replyid)

def timestring(minutes):
    h, m = divmod(minutes, 60)
    d, h = divmod(h, 24)
    return (' %d 天' % d if d else '') + (' %d 小时' % h if h else '') + (' %d 分钟' % m if m else '')

def cmd_uinfo(expr, chatid, replyid, msg):
    '''/user|/uinfo [@username] [minutes=1440] Show information about <@username>.'''
    if 'reply_to_message' in msg:
        uid = msg['reply_to_message']['from']['id']
    else:
        uid = None
    if expr:
        expr = expr.split(' ')
        username = expr[0]
        if not username.startswith('@'):
            uid = uid or msg['from']['id']
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
        uid = uid or msg['from']['id']
        minutes = 1440
    user = db_getuser(uid)
    uinfoln = []
    if user[0]:
        uinfoln.append('@' + user[0])
    uinfoln.append(db_getufname(uid))
    uinfoln.append('ID: %s' % uid)
    result = [', '.join(uinfoln)]
    if msg['chat']['id'] == -CFG['groupid']:
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
    msg = ['在最近%s内有 %s 条消息，平均每分钟 %.2f 条。' % (timestr, count, count/minutes)]
    msg.extend('%s: %s 条，%.2f%%' % (db_getufname(k), v, v/count*100) for k, v in mcomm)
    msg.append('其他用户 %s 条，人均 %.2f 条' % (count - sum(v for k, v in mcomm), count / len(ctr)))
    sendmsg('\n'.join(msg), chatid, replyid)

def cmd_digest(expr, chatid, replyid, msg):
    sendmsg('Not implemented.', chatid, replyid)

def cmd_calc(expr, chatid, replyid, msg):
    '''/calc <expr> Calculate <expr>.'''
    if expr:
        runapptask('calc', (expr,), (chatid, replyid))
    else:
        sendmsg('Syntax error. Usage: ' + cmd_calc.__doc__, chatid, replyid)

def cmd_py(expr, chatid, replyid, msg):
    '''/py <expr> Evaluate Python 2 expression <expr>.'''
    if expr:
        if len(expr) > 1000:
            sendmsg('Expression too long.', chatid, replyid)
        else:
            runapptask('py', (expr,), (chatid, replyid))
    else:
        sendmsg('Syntax error. Usage: ' + cmd_py.__doc__, chatid, replyid)

def cmd_bf(expr, chatid, replyid, msg):
    '''/bf <expr> [|<input>] Evaluate Brainf*ck expression <expr> (with <input>).'''
    if expr:
        expr = expr.split('|', 1)
        inpt = expr[1] if len(expr) > 1 else ''
        runapptask('bf', (expr[0], inpt), (chatid, replyid))
    else:
        sendmsg('Syntax error. Usage: ' + cmd_bf.__doc__, chatid, replyid)

def cmd_lisp(expr, chatid, replyid, msg):
    '''/lisp <expr> Evaluate Lisp(Scheme)-like expression <expr>.'''
    if expr:
        runapptask('lisp', (expr,), (chatid, replyid))
    else:
        sendmsg('Syntax error. Usage: ' + cmd_lisp.__doc__, chatid, replyid)

def cmd_name(expr, chatid, replyid, msg):
    '''/name [pinyin] Get a Chinese name.'''
    runapptask('name', (expr,), (chatid, replyid))

def cmd_cc(expr, chatid, replyid, msg):
    '''/cc <Chinese> Simplified-Traditional Chinese conversion.'''
    tinput = ''
    if 'reply_to_message' in msg:
        tinput = msg['reply_to_message'].get('text', '')
    tinput = (expr or tinput).strip()
    runapptask('cc', (tinput,), (chatid, replyid))

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
    runapptask('ime', (tinput,), (chatid, replyid))

def cmd_cut(expr, chatid, replyid, msg):
    '''/cut [c|m] <something> Segment <something>.'''
    if expr[:2].strip() == 'c':
        lang = 'c'
        expr = expr[2:]
    elif expr[:2].strip() == 'm':
        lang = 'm'
        expr = expr[2:]
    else:
        lang = None
    tinput = ''
    if 'reply_to_message' in msg:
        tinput = msg['reply_to_message'].get('text', '')
    tinput = (expr or tinput).strip()
    if len(tinput) > 1000:
        tinput = tinput[:1000] + '……'
    if not tinput:
        sendmsg('Syntax error. Usage: ' + cmd_cut.__doc__, chatid, replyid)
        return
    runapptask('cut', (tinput, lang), (chatid, replyid))

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
    if len(tinput) > 1000:
        tinput = tinput[:1000] + '……'
    if not tinput:
        sendmsg('Syntax error. Usage: ' + cmd_wyw.__doc__, chatid, replyid)
        return
    typing(chatid)
    runapptask('wyw', (tinput, lang), (chatid, replyid))

def cmd_say(expr, chatid, replyid, msg):
    '''/say Say something interesting.'''
    #typing(chatid)
    if expr:
        runapptask('reply', (expr,), (chatid, replyid))
    else:
        runapptask('say', (), (chatid, replyid))

def cmd_mgw(expr, chatid, replyid, msg):
    if chatid < 0:
        return
    runapptask('mgw', (), (chatid, replyid))

def cmd_reply(expr, chatid, replyid, msg):
    '''/reply [question] Reply to the conversation.'''
    if 'forward_from' in msg and msg['chat']['id'] < 0:
        return
    typing(chatid)
    text = ''
    if 'reply_to_message' in msg:
        text = msg['reply_to_message'].get('text', '')
    text = (expr.strip() or text or ' '.join(t[0] for t in conn.execute("SELECT text FROM messages ORDER BY date DESC LIMIT 2").fetchall())).replace('\n', ' ')
    runapptask('reply', (text,), (chatid, replyid))

def cmd_cont(expr, chatid, replyid, msg):
    '''/cont [sentence] Complete the sentence.'''
    if 'forward_from' in msg and msg['chat']['id'] < 0:
        return
    typing(chatid)
    text = ''
    if 'reply_to_message' in msg:
        text = msg['reply_to_message'].get('text', '')
    text = (expr.strip() or text or conn.execute("SELECT text FROM messages ORDER BY date DESC LIMIT 1").fetchone()[0]).replace('\n', ' ')
    runapptask('cont', (text,), (chatid, replyid))

def cmd_echo(expr, chatid, replyid, msg):
    '''/echo Parrot back.'''
    if 'ping' in expr.lower():
        sendmsg('pong', chatid, replyid)
    elif expr:
        sendmsg(expr, chatid, replyid)
    else:
        sendmsg('ping', chatid, replyid)

def cmd_do(expr, chatid, replyid, msg):
    actions = collections.OrderedDict((
        ('shrug', '¯\\_(ツ)_/¯'),
        ('lenny', '( ͡° ͜ʖ ͡°)'),
        ('flip', '（╯°□°）╯︵ ┻━┻'),
        ('homo', '┌（┌　＾o＾）┐'),
        ('look', 'ಠ_ಠ'),
        ('cn', '[citation needed]'),
        ('boom', '💥'),
        ('tweet', '🐦'),
        ('blink', '👀'),
        ('see-no-evil', '🙈'),
        ('hear-no-evil', '🙉'),
        ('speak-no-evil', '🙊'),
        ('however', ('不要怪我们没有警告过你\n我们都有不顺利的时候\n'
                     'Something happened\n这真是让人尴尬\n'
                     '请坐和放宽，滚回以前的版本\n这就是你的人生\n是的，你的人生'))
    ))
    expr = expr.lower()
    res = actions.get(expr)
    if res:
        sendmsg(res, chatid, replyid)
    elif expr == 'help':
        sendmsg(', '.join(actions.keys()), chatid, replyid)
    else:
        try:
            res = unicodedata.lookup(expr)
            sendmsg(res, chatid, replyid)
            return
        except KeyError:
            pass
        if len(expr) == 1:
            try:
                res = unicodedata.name(expr)
                sendmsg(res, chatid, replyid)
            except ValueError:
                sendmsg('Character not found in Unicode %s' % unicodedata.unidata_version, chatid, replyid)
        else:
            sendmsg('Something happened.', chatid, replyid)

def cmd_t2i(expr, chatid, replyid, msg):
    global CFG
    if msg['chat']['id'] == -CFG['groupid']:
        if expr == 'off' or CFG.get('t2i'):
            CFG['t2i'] = False
            sendmsg('Telegram to IRC forwarding disabled.', chatid, replyid)
        elif expr == 'on' or not CFG.get('t2i'):
            CFG['t2i'] = True
            sendmsg('Telegram to IRC forwarding enabled.', chatid, replyid)

def cmd_i2t(expr, chatid, replyid, msg):
    global CFG
    if msg['chat']['id'] == -CFG['groupid']:
        if expr == 'off' or CFG.get('i2t'):
            CFG['i2t'] = False
            sendmsg('IRC to Telegram forwarding disabled.', chatid, replyid)
        elif expr == 'on' or not CFG.get('i2t'):
            CFG['i2t'] = True
            sendmsg('IRC to Telegram forwarding enabled.', chatid, replyid)

def cmd_autoclose(expr, chatid, replyid, msg):
    global CFG
    if msg['chat']['id'] == -CFG['groupid']:
        if CFG.get('autoclose'):
            CFG['autoclose'] = False
            sendmsg('Auto closing brackets disabled.', chatid, replyid)
        else:
            CFG['autoclose'] = True
            sendmsg('Auto closing brackets enabled.', chatid, replyid)

def cmd_cancel(expr, chatid, replyid, msg):
    '''/cancel Hide keyboard and interrupt current session.'''
    bot_api('sendMessage', chat_id=chatid, text='Cancelled.', reply_to_message_id=replyid, reply_markup='{"hide_keyboard": true}')

def cmd__cmd(expr, chatid, replyid, msg):
    global SAY_P, APP_P
    if chatid < 0:
        return
    if expr == 'killserver':
        APP_P.terminate()
        APP_P = subprocess.Popen(APP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        checkappproc()
        sendmsg('Server restarted.', chatid, replyid)
        logging.info('Server restarted upon user request.')
    elif expr == 'commit':
        while 1:
            try:
                logmsg(LOG_Q.get_nowait())
            except queue.Empty:
                break
        db.commit()
        sendmsg('DB committed.', chatid, replyid)
        logging.info('DB committed upon user request.')
    #elif expr == 'raiseex':  # For debug
        #async_func(_raise_ex)(Exception('/_cmd raiseex'))
    #else:
        #sendmsg('ping', chatid, replyid)

def cmd__welcome(expr, chatid, replyid, msg):
    if chatid > 0:
        return
    usr = msg["new_chat_participant"]
    USER_CACHE[usr["id"]] = (usr.get("username"), usr.get("first_name"), usr.get("last_name"))
    sendmsg('欢迎 %s 加入本群！' % dc_getufname(usr), chatid, replyid)

facescore = lambda x,y: 1/2*math.erfc((0.5*y-x)/(2**0.5*(0.5*y**0.5)))*100

fstable = [facescore(i, 100) for i in range(101)]
revface = lambda x: min((abs(x-v), k) for k,v in enumerate(fstable))[1]

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
        txt += '\n' + '(🌝%d/🌚%d' % (wcount, num - wcount)
        if num > 41:
            txt += ', 🌝%.2f%%' % facescore(wcount, num)
        txt += ')'
    sendmsg(txt, chatid, replyid)

def cmd_fig(expr, chatid, replyid, msg):
    '''/fig <char> Make figure out of moon faces.'''
    if expr:
        runapptask('fig', (expr,), (chatid, replyid))
    else:
        sendmsg(srandom.choice('🌝🌚'), chatid, replyid)

def cmd_start(expr, chatid, replyid, msg):
    if chatid != -CFG['groupid']:
        sendmsg('This is Orz Digger. It can help you search the long and boring chat log of the ##Orz group.\nSend me /help for help.', chatid, replyid)

def cmd_help(expr, chatid, replyid, msg):
    '''/help Show usage.'''
    if expr:
        if expr in COMMANDS:
            h = COMMANDS[expr].__doc__
            if h:
                sendmsg(h, chatid, replyid)
            else:
                sendmsg('Help is not available for ' + expr, chatid, replyid)
        else:
            sendmsg('Command not found.', chatid, replyid)
    elif chatid == -CFG['groupid']:
        sendmsg('Full help disabled in this group.', chatid, replyid)
    elif chatid > 0:
        sendmsg('\n'.join(uniq(cmd.__doc__ for cmd in COMMANDS.values() if cmd.__doc__)), chatid, replyid)
    else:
        sendmsg('\n'.join(uniq(cmd.__doc__ for cmdname, cmd in COMMANDS.items() if cmd.__doc__ and cmdname in PUBLIC)), chatid, replyid)

def sig_commit(signum, frame):
    db.commit()
    logging.info('DB committed upon signal %s' % signum)

# should document usage in docstrings
COMMANDS = collections.OrderedDict((
('m', cmd_getmsg),
('context', cmd_context),
('s', cmd_search),
('search', cmd_search),
('mention', cmd_mention),
('user', cmd_uinfo),
('uinfo', cmd_uinfo),
('digest', cmd_digest),
('stat', cmd_stat),
('calc', cmd_calc),
#('calc', cmd_py),
('py', cmd_py),
('bf', cmd_bf),
('lisp', cmd_lisp),
('name', cmd_name),
('ime', cmd_ime),
('fig', cmd_fig),
('cc', cmd_cc),
('quote', cmd_quote),
('wyw', cmd_wyw),
('cut', cmd_cut),
('mgw', cmd_mgw),
('say', cmd_say),
('reply', cmd_reply),
#('cont', cmd_cont),
#('echo', cmd_echo),
('do', cmd_do),
('t2i', cmd_t2i),
('i2t', cmd_i2t),
('autoclose', cmd_autoclose),
('233', cmd_233),
('start', cmd_start),
('help', cmd_help),
('cancel', cmd_cancel),
('_cmd', cmd__cmd)
))

PUBLIC = set((
'user',
'calc',
'py',
'bf',
'lisp',
'name',
'ime',
'fig',
'cc',
'wyw',
'cut',
'say',
'reply',
#'cont',
#'echo',
'do',
'233',
'start',
'cancel',
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
URL_FILE = 'https://api.telegram.org/file/bot%s/' % CFG['token']

# Initialize messages in database

#importdb('telegram-history.db')
#importupdates(OFFSET, 2000)
#importfixservice('telegram-history.db')
#sys.exit(0)

signal.signal(signal.SIGUSR1, sig_commit)

MSG_Q = queue.Queue()
LOG_Q = queue.Queue()
APP_TASK = {}
APP_LCK = threading.Lock()
APP_CMD = ('python3', 'appserve.py')
APP_P = subprocess.Popen(APP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
executor = concurrent.futures.ThreadPoolExecutor(10)

pollthr = threading.Thread(target=getupdates)
pollthr.daemon = True
pollthr.start()

appthr = threading.Thread(target=getappresult)
appthr.daemon = True
appthr.start()

ircconn = None
if 'ircserver' in CFG:
    checkircconn()
    ircthr = threading.Thread(target=getircupd)
    ircthr.daemon = True
    ircthr.start()

# fx233es = fparser.Parser(numtype='decimal')

logging.info('Satellite launched.')

try:
    while 1:
        try:
            processmsg()
        except Exception as ex:
            logging.exception('Failed to process a message.')
            continue
finally:
    while 1:
        try:
            logmsg(LOG_Q.get_nowait())
        except queue.Empty:
            break
    conn.execute('REPLACE INTO config (id, val) VALUES (0, ?)', (OFFSET,))
    conn.execute('REPLACE INTO config (id, val) VALUES (1, ?)', (IRCOFFSET,))
    json.dump(CFG, open('config.json', 'w'), sort_keys=True, indent=4)
    db.commit()
    APP_P.terminate()
    logging.info('Shut down cleanly.')
