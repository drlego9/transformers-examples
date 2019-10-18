#!/bin/zsh
echo $SHELL
echo "Extracting from Korean Wikipedia data..."
python distillation-kor/scripts/wiki_extractor.py
    --input .data/wiki-kor/kowiki-20190901-pages-articles-multistream.xml \
    --json \
    --processes 4 \
    --output ".data/wiki-kor-extracted/" \
    --log_file ".logs/run_wiki_extractor.log"
echo "Finished."