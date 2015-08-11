# tg-chatdig
Dig into long and boring Telegram group chat logs.

## chatdig.py

Main script, handles a lot of commands. Uses a SQLite 3 database to store messages.

## tglog-import.py

Executes `telegram-cli` and fetches history messages.

## vendor/

Some interesting functions.

### say.py

Randomly writes out sentences according to the language model.

Depends on [jieba](https://github.com/fxsjy/jieba), [kenlm](https://github.com/kpu/kenlm).

See `vendor/updatelm.sh` for building language models.

### seccomp.py

Evals user input and prints out result safely. Originally written by David Wison.

See [dw/scratch/seccomp.py](https://github.com/dw/scratch/blob/master/seccomp.py)

### fparser.py

See [gumblex/fxcalc](https://github.com/gumblex/fxcalc)

### External Plugins

The following components are not in this repo:

* `/bf` bf.py: [Brainf*ck interpreter](http://www.cs.princeton.edu/~ynaamad/misc/bf.htm)
* `/lisp` lispy.py: [Scheme-like interpreter](http://norvig.com/lispy.html)
* `/name` , namemodel.m: Part of [Chinese name generator](https://github.com/gumblex/chinesename)
* `/ime` simpleime.py, pinyinlookup.py, \*.dawg: [Simple Pinyin IME](https://github.com/gumblex/simpleime)
* zhconv.py, zhcdict.json: [Simplified-Traditional Chinese converter](https://github.com/gumblex/zhconv)
