#!/bin/bash

#### Edit paths before using

sqlite3 ../chatlog.db <<< 'select text from messages where text is not null and src != 120400693;' | perl -p -e 's|^[^\n ]+] ||' | python3 logcutfilter.py | opencc -c t2s.json | awk '!seen[$0]++' > chatlogf.txt

sed 's/ /\n/g' chatlogf.txt | awk '{seen[$0]++} END {for (i in seen) {if (seen[i] > 1) print i}}' > chatdict.txt

/media/E/corpus/moses/bin/lmplz -o 6 --prune 0 0 0 0 0 1 -S 50% --text chatlogf.txt --arpa chat.lm
/media/E/corpus/moses/bin/build_binary trie -q 8 -b 8 chat.lm chat.binlm

pypy3 learnctx.py chatdict.txt < chatlogf.txt
