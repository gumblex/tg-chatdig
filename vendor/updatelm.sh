#!/bin/bash

#### Edit paths before using

sqlite3 ../chatlog.db <<< 'select text from messages where text is not null and text != "" and text not like "/%" and src != 120400693;' | tee chatlog.txt | python3 ../truecaser.py -t truecase.txt
pv chatlog.txt | python3 ../truecaser.py truecase.txt | perl -p -e 's|^[^\n ]+] ||' | python3 logcutfilter.py | opencc -c t2s.json | awk '!seen[$0]++' | tee chatlogf.txt | sed 's/“//g;s/”//g;s/  / /g;s/ /\n/g' | awk '{seen[$0]++} END {for (i in seen) {if (seen[i] > 5) print i}}' > chatdict.txt
rm chatlog.txt

~/software/moses/bin/lmplz -o 6 --prune 0 0 0 0 0 1 -S 50% --text chatlogf.txt --arpa chat.lm
~/software/moses/bin/build_binary trie chat.lm chat.binlm

rm chat.lm
pv chatlogf.txt | pypy3 learnctx.py chatdict.txt
