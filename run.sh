#!/bin/bash

PROBLEM=algorithmic_addition_binary40
PROBLEM=translate_enfr_wmt_small8k
MODEL=universal_transformer_mohd
HPARAMS=universal_transformer_mohd_hparams

DATA_DIR=./t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=./t2t_train/

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-trainer \
    --worker_gpu=0 \
    --t2t_usr_dir=. \
    --generate_data \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=1000 \
    --eval_steps=100
