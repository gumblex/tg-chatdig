#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import requests
from .base import *

socket.setdefaulttimeout(60)

class TelegramBotAPIFailed(Exception):
    pass

class TelegramBotProtocal(MessageProtocal):

    name = 'tgbot'

    def setup(self, host):
        self.host = host
        self.offset = host.sqlite.execute('SELECT val FROM config WHERE id = 0').fetchone()
        self.offset = self.offset[0] if self.offset else 0
        self.HSession = requests.Session()
        self.useragent = 'TgChatDiggerBot/%s %s' % (__version__, self.HSession.headers["User-Agent"])
        self.HSession.headers["User-Agent"] = useragent
        self.msg_cache = LRUCache(20)
        self.user_cache = LRUCache(10)

    def bot_api(method, **params):
        for att in range(3):
            try:
                req = self.HSession.get(URL + method, params=params, timeout=45)
                retjson = req.content
                ret = json.loads(retjson.decode('utf-8'))
                break
            except Exception as ex:
                if att < 1:
                    time.sleep((att+1) * 2)
                else:
                    raise ex
        if not ret['ok']:
            raise TelegramBotAPIFailed(repr(ret))
        return ret['result']

    def run(self):
        while 1:
            try:
                updates = bot_api('getUpdates', offset=self.offset, timeout=10)
            except Exception as ex:
                logging.exception('Get updates failed.')
                continue
            if updates:
                logging.debug('Messages coming.')
                self.offset = updates[-1]["update_id"] + 1
                for upd in updates:
                    self.processmsg(upd)
            time.sleep(.2)

    def classify(self, msg):
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

    @async_method
    def processmsg(self, d):
        logging.debug('Msg arrived: %r' % d)
        uid = d['update_id']
        if 'message' in d:
            msg = d['message']
            if 'text' in msg:
                msg['text'] = msg['text'].replace('\xa0', ' ')
            elif 'caption' in msg:
                msg['text'] = msg['caption'].replace('\xa0', ' ')
            self.msg_cache[msg['message_id']] = msg
            cls = classify(msg)
            logging.debug('Classified as: %s', cls)
            self.host.newmsg(Message(msg, self.name, i=cls))
