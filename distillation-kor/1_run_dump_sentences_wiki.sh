#!/bin/zsh
echo "Dumping kowiki data, one sentence per line..."
python distillation-kor/scripts/dump_sentences_wiki.py \
    --root '.data/wiki-kor-extracted/' \
    --dump_file '.data/dump_wiki-kor.txt'
echo "Finished."