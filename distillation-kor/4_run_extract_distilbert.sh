#!/bin/bash
echo "Initializing student model weights."

MODEL_TYPE='kobert'
MODEL_NAME='kobert-8002'
PRETRAINED_DIR='.kobert/pretrained/'

echo 'Do you want vocab transformation? (y/n)'
read doVocabTransform 

if [ $doVocabTransform = 'y' ]
    then
    python distillation-kor/scripts/extract_distilbert.py \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --pretrained_dir $PRETRAINED_DIR \
        --dump_checkpoint ".kobert/tf_${MODEL_NAME}_vocab_transformed.pth" \
        --vocab_transform
else
    python distillation-kor/scripts/extract_distilbert.py \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --pretrained_dir $PRETRAINED_DIR \
        --dump_checkpoint ".kobert/tf_${MODEL_NAME}.pth"
fi

echo "Finished retrieving initial weights for the student model from ${MODEL_NAME}"
