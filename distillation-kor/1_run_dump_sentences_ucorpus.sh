#!/bin/bash
echo "Dumping Exbrain UCorpus data, one sentence per line..."
python distillation-kor/scripts/dump_sentences_ucorpus.py \
    --root '.data/exbrain-ucorpus/' \
    --dump_file '.data/dump_ucorpus.txt'
echo "Finished."