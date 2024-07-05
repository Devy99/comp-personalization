#!/bin/bash

#### This script generates the prediction of all the stored checkpoints and find the best one ####

# Variable related to the evaluation dataset 
DATASET_EVAL="../3-general-datasets/datasets/finetuning/finetuning_methods_eval.csv"

# Names of the columns containing inputs and outputs to feed the model
INPUT_COLNAME="masked_method"
TARGET_COLNAME="mask"

# Directory containing the tokenizer .model file 
TOKENIZER_MODEL="./T5_Configs/tokenizer.model"

# Directory containing the saved checkpoints 
CHECKPOINTS_DIR="./checkpoints"

# Name of the TXT file containing the predictions, stored in each checkpoint
PREDICTIONS_FILENAME="predictions_eval.txt"

# Update jax according to the Conda version
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# List all the checkpoints and generate the predictions
directories=( $(ls -d $CHECKPOINTS_DIR/*/ | sort --version-sort ) )
for entry in "${directories[@]}"
do
  echo "Generating predictions for checkpoint located in: $entry"
  
  # Launch single prediction on the relative checkpoint
  python infer.py \
      --model_name_or_path="$entry"  \
      --tokenizer_name="$TOKENIZER_MODEL" \
      --source_column="$INPUT_COLNAME" \
      --target_column="$TARGET_COLNAME" \
      --max_source_length="1024" \
      --max_target_length="256" \
      --use_fast_tokenizer \
      --dataset_path="$DATASET_EVAL" \
      --dataset_split="validation" \
      --batch_size="64" \
      --output_dir="$entry" \
      --predictions_filename="$PREDICTIONS_FILENAME"
done

# Get best checkpoint
python get_best_checkpoint.py \
    --eval_file="$DATASET_EVAL"  \
    --checkpoints_dir="$CHECKPOINTS_DIR" \
    --predictions_filename="$PREDICTIONS_FILENAME"  \
    --target_colname="$TARGET_COLNAME" 