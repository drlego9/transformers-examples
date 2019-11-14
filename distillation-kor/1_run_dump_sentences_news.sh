#!/bin/bash
echo "Dumping Korean news data, one sentence per line..."
python distillation-kor/scripts/dump_sentences_news.py \
    --root '.data/news-kor/' \
    --dump_file '.data/dump_news-kor.txt'
echo "Finished."