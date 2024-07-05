#!/bin/bash

# Variables related to the dataset splits
DATASET_TRAIN='../3-general-datasets/datasets/pretraining/pretraining_train.txt'
DATASET_EVAL='../3-general-datasets/datasets/pretraining/pretraining_eval.txt'

# Directories that will contain the pre-trained checkpoints and the model configuration / tokenizer
OUTPUT_DIR='./checkpoints'
CONFIG_DIR='./T5_Configs'
MAX_TOKENS_LENGTH='1024'
TOKENIZER_MODEL='./T5_Configs/tokenizer.model' # generated automatically from tokenizer_config_gen.py

# Generate output directories
mkdir $OUTPUT_DIR
mkdir $CONFIG_DIR

# Train sentencepiece tokenizer
python ./utils/tokenizer.py \
        --input="$DATASET_TRAIN" \
        --folder_name="$CONFIG_DIR" \
        --max_tokens_length="$MAX_TOKENS_LENGTH" 

# Copy configurations in fine-tuning folders
FINETUNING_DIR='../fine-tuning'
cp -r $CONFIG_DIR $FINETUNING_DIR

# Update jax according to the Conda version
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Run pre-training
python ./utils/run_t5_mlm_flax.py \
        --output_dir="$OUTPUT_DIR" \
        --model_type="t5" \
        --config_name="$CONFIG_DIR" \
        --tokenizer_name="$TOKENIZER_MODEL" \
        --do_train \
        --use_fast_tokenizer \
        --train_file="$DATASET_TRAIN" \
        --validation_file="$DATASET_EVAL" \
        --max_seq_length="$MAX_TOKENS_LENGTH" \
        --per_device_train_batch_size="16" \
        --per_device_eval_batch_size="16" \
        --adafactor \
        --learning_rate="0.005" \
        --weight_decay="0.001" \
        --warmup_steps="2000" \
        --overwrite_output_dir \
        --logging_steps="500" \
        --save_steps="5000" \
        --eval_steps="5000" \
        --num_train_epochs="50"

