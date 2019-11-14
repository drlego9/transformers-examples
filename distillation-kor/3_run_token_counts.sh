#!/bin/bash
echo "Counting token occurences."
python distillation-kor/scripts/token_counts.py \
    --data_file ".data/dump_merged.kobert-8002.pickle" \
    --token_counts_dump ".data/token_counts.kobert-8002.pickle" \
    --vocab_size 8002
echo "Finished."