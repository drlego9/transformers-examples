#!/bin/bash
python distillation-kor/scripts/binarized_data.py \
    --file_path ".data/dump_merged.txt" \
    --tokenizer_type "kobert" \
    --tokenizer_name "kobert-8002" \
    --dump_file './.data/dump_merged'
echo "Finished."