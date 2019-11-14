#!/bin/bash
echo "Merging multiple dump files."
python distillation-kor/scripts/merge_dumps.py \
    --dump_root '.data/' \
    --in_files 'dump_wiki-kor.txt' 'dump_aihub.txt' 'dump_ucorpus.txt' 'dump_news-kor.txt' \
    --out_file 'dump_merged.txt'
echo "Finished."