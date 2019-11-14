#!/bin/bash
python distillation-kor/scripts/binarized_data.py \
    --file_path ".data/dump_wiki-kor.txt" \
    --tokenizer_type "bert" \
    --tokenizer_name "bert-base-multilingual-cased" \
    --dump_file './.data/dump_wiki-kor'
echo "Finished."
