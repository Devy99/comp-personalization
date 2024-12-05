#!/bin/bash

NUM_EPOCHS=10
BATCH_SIZE=$4

MAX_SOURCE_LENGTH=2304
MAX_TARGET_LENGTH=256

OUTPUT_DIR="${3}/checkpoints-codellama-7b-lora-dev-${1}"
HF_TOKEN="<your_token>"

DEVELOPER_NUM=$1
TRAIN_FILENAME=$2 # developer_masked_methods_train.json or apache_dataset_total_train.json

ACCELERATE_CONFIG=$5

# Run the training script
accelerate launch --config_file "$ACCELERATE_CONFIG" codellama_lora_finetuning.py \
    --model_name_or_path="meta-llama/CodeLlama-7b-hf" \
    --hf_token=$HF_TOKEN \
    --output_dir=$OUTPUT_DIR \
    --train_data="datasets_parsed/developer_${DEVELOPER_NUM}/${TRAIN_FILENAME}" \
    --max_source_len=$MAX_SOURCE_LENGTH \
    --max_target_len=$MAX_TARGET_LENGTH \
    --source_column="parsed_masked_method" \
    --target_column="parsed_mask" \
    --batch_size=$BATCH_SIZE \
    --epochs=$NUM_EPOCHS