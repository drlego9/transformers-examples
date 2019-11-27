#!/bin/bash
# Training on distributed is ok.
# Evaluation on distributed is NOT ok. Use single GPU only.

echo "Current Working Directory: ${PWD}"

FINETUNED_MODEL_TYPE="distilkobert"
FINETUNED_MODEL_DIR="./distilkobert/kobert-8002/finetune_outputs/"
FINETUNED_MODEL_CONFIG_PATH="${FINETUNED_MODEL_DIR}/config.json"
TOKENIZER_VOCAB_PATH="${FINETUNED_MODEL_DIR}/vocab.txt"
EVALUATION_OUTPUT_DIR="${FINETUNED_MODEL_DIR}/evaluation_outputs/"
mkdir -p ${EVALUATION_OUTPUT_DIR}

TRAIN_FILE=".data/korquad/KorQuAD_v1.0_train.json"
PREDICT_FILE=".data/korquad/KorQuAD_v1.0_dev.json"

# Model hyperparameters
MAX_SEQ_LENGTH=384
DOC_STRIDE=128
PER_GPU_TRAIN_BATCH_SIZE=3
PER_GPU_EVAL_BATCH_SIZE=3

# Kill existing process (may not be necessary)
pkill -f "python -u run_squad.py"

python run_squad.py \
    --do_eval \
    --model_type $FINETUNED_MODEL_TYPE \
    --model_name_or_path $FINETUNED_MODEL_DIR \
    --config_name $FINETUNED_MODEL_CONFIG_PATH \
    --tokenizer_name $TOKENIZER_VOCAB_PATH \
    --train_file $TRAIN_FILE \
    --predict_file $PREDICT_FILE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --doc_stride $DOC_STRIDE \
    --output_dir $EVALUATION_OUTPUT_DIR \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE
