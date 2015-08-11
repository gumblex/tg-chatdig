# tg-chatdig
Dig into long and boring Telegram group chat logs.

## chatdig.py

Main script, handles a lot of commands. Uses a SQLite 3 database to store messages.

## tglog-import.py

Executes telegram-cli and fetches history messages.

## vendor/

Some interesting functions.

### say.py

Randomly writes out sentences according to the language model.

Depends on [jieba](https://github.com/fxsjy/jieba), [kenlm](https://github.com/kpu/kenlm).

### seccomp.py

Evals user input and prints out result safely. Originally written by David Wison.

See https://github.com/dw/scratch/blob/master/seccomp.py

### fparser.py

See https://github.com/gumblex/fxcalc

