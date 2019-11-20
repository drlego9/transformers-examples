#!/bin/bash
# Training on distributed is ok.
# Evaluation on distributed is NOT ok. Use single GPU only.

echo "Current Working Directory: ${PWD}"

TEACHER_ROOT=".kobert"
STUDENT_ROOT=".distilkobert/kobert-8002"

STUDENT_MODEL_TYPE="distilkobert"
STUDENT_MODEL_DIR="${STUDENT_ROOT}/finetune_inputs/"
FINETUNE_OUTPUT_DIR="${STUDENT_ROOT}/finetune_outputs/"
MODEL_CONFIG_PATH="${STUDENT_ROOT}/config.json"
TOKENIZER_CONFIG_PATH="${TEACHER_ROOT}/kobert_tokenizer_config.json"
TRAIN_FILE=".data/korquad/KorQuAD_v1.0_train.json"
PREDICT_FILE=".data/korquad/KorQuAD_v1.0_dev.json"

# Model train / evaluation hyperparameters
LEARNING_RATE=3e-5
EPOCHS=3
MAX_SEQ_LENGTH=384
DOC_STRIDE=128
PER_GPU_TRAIN_BATCH_SIZE=3
PER_GPU_EVAL_BATCH_SIZE=3

echo "Teacher root directory: ${TEACHER_ROOT}/"
echo "Student root directory: ${STUDENT_ROOT}/"

ls ${STUDENT_ROOT}
echo "Select student model number (0, 1, ...):"
read MODEL_NUMBER

mkdir -p ${STUDENT_MODEL_DIR}
cp "${STUDENT_ROOT}/model_epoch_${MODEL_NUMBER}.pth" "${STUDENT_MODEL_DIR}/pytorch_model.bin"
cp "${STUDENT_ROOT}/config.json" "${STUDENT_MODEL_DIR}/config.json"

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    run_squad.py \
        --model_type $STUDENT_MODEL_TYPE \
        --model_name_or_path $STUDENT_MODEL_DIR \
        --config_name $MODEL_CONFIG_PATH \
        --tokenizer_name $TOKENIZER_CONFIG_PATH \
        --do_train \
        --train_file $TRAIN_FILE \
        --predict_file $PREDICT_FILE \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $EPOCHS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --doc_stride $DOC_STRIDE \
        --output_dir $FINETUNE_OUTPUT_DIR \
        --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
        --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE
