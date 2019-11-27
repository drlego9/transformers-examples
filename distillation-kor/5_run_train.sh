#!/bin/bash
echo "Distributed training of DistilKoBERT."

export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_PORT=31415
export MASTER_ADDR=127.0.0.1

N_EPOCH=20
TEACHER_TYPE='kobert'
TEACHER_NAME='kobert-8002'
STUDENT_TYPE='distilkobert'
STUDENT_CONFIG_FILE='.kobert/distilkobert_student_config.json'
STUDENT_PRETRAINED_WEIGHTS_PATH='.kobert/tf_kobert-8002_vocab_transformed.pth'
DUMP_PATH='.distilkobert/kobert-8002/'
DATA_FILE='.data/dump_merged.kobert-8002.pickle'
TOKEN_COUNTS_FILE='.data/token_counts.kobert-8002.pickle'

pkill -f "python -u distillation-kor/train.py"

python -u -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    distillation-kor/train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --n_epoch $N_EPOCH \
        --student_type $STUDENT_TYPE \
        --student_config $STUDENT_CONFIG_FILE \
        --student_pretrained_weights $STUDENT_PRETRAINED_WEIGHTS_PATH \
        --teacher_type $TEACHER_TYPE \
        --teacher_name $TEACHER_NAME \
        --dump_path $DUMP_PATH \
        --data_file $DATA_FILE \
        --token_counts $TOKEN_COUNTS_FILE \
        --alpha_ce 0.33 \
        --alpha_mlm 0.33 \
        --alpha_clm 0.00 \
        --alpha_cos 0.33 \
        --mlm
