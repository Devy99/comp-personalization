#!/bin/bash

# Variables related to the training dataset
DATASET_TRAIN="../3-general-datasets/datasets/finetuning/finetuning_methods_train.csv"

# Names of the columns containing inputs and outputs to feed the model
INPUT_COLNAME="masked_method"
TARGET_COLNAME="mask"
MAX_TOKENS_LENGTH="1024"

# Pretrained checkpoint, directory containing the tokenizer JSON file and
# output directory where to save fine-tuning checkpoints
PRETRAINING_DIR="./ckpt-pretraining"
TOKENIZER_MODEL='./T5_Configs/tokenizer.model'
OUTPUT_DIR="./checkpoints"

# Generate the output directory
mkdir $OUTPUT_DIR

# Update jax according to the Conda version
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Launch finetuning
python finetuning.py \
    --model_name_or_path="$PRETRAINING_DIR"  \
    --tokenizer_name="$TOKENIZER_MODEL" \
    --from_flax \
    --do_train \
    --use_fast_tokenizer \
    --train_file="$DATASET_TRAIN" \
    --source_column="$INPUT_COLNAME" \
    --target_column="$TARGET_COLNAME" \
    --max_source_length="$MAX_TOKENS_LENGTH" \
    --save_strategy="epoch" \
    --logging_steps="10000" \
    --num_train_epochs="10"  \
    --save_total_limit="1000" \
    --output_dir="$OUTPUT_DIR" \
    --per_device_train_batch_size="48" \
    --fp16
#   --resume_training Uncomment if you want to resume the training