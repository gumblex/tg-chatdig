#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import cgi
import json
import sqlite3
import calendar
from email.utils import formatdate, parsedate

DB_FILE = 'chatlog.db'

MTIME = os.path.getmtime(DB_FILE)
CONN = sqlite3.connect(DB_FILE)

def auth(sqltype, arg1, arg2, dbname, source):
    if sqltype in (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    else:
        return sqlite3.SQLITE_DENY

def do_query(form):
    try:
        sql = form['q'].value
        cur = CONN.cursor()
        cur.execute(sql)
        return '200 OK', json.dumps({
            'ret': 200,
            'description': [desc[0] for desc in cur.description],
            'rows': cur.fetchall()
        }).encode('utf-8')
    except Exception as ex:
        return '400 Bad Request', json.dumps({
            'ret': 400,
            'error': str(ex)
        }).encode('utf-8')

form = cgi.FieldStorage()
try:
    if calendar.timegm(parsedate(os.environ['HTTP_IF_MODIFIED_SINCE'])) >= MTIME:
        print("Status: 304 Not Modified")
        print()
        sys.exit(0)
except Exception:
    pass

status, reply = do_query(form)
print("Status: " + status)
print("Content-Type: application/json; charset=utf-8")
print("Content-Length: %d" % len(reply))
print("Last-Modified: " + formatdate(MTIME, False, True))
print("Connection: close")
print()
sys.stdout.flush()
sys.stdout.buffer.write(reply)
sys.stdout.buffer.flush()
