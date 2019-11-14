#!/bin/bash
echo "Dumping Korean text from AIHub translation data, one sentence per line..."
python distillation-kor/scripts/dump_sentences_aihub.py \
    --root '.data/aihub-koen/' \
    --dump_file '.data/dump_aihub.txt'
echo "Finished."