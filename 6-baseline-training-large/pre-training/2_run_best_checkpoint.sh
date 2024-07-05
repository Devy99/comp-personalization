#!/bin/bash

#### This script generates the evaluation of all the stored checkpoints to find the best one ####
DATASET_TRAIN='../3-general-datasets/datasets/pretraining/pretraining_train.txt'
DATASET_EVAL='../3-general-datasets/datasets/pretraining/pretraining_eval.txt'

OUTPUT_DIR='./checkpoints'
CONFIG_DIR='./T5_Configs'

MAX_TOKENS_LENGTH='1024'
TOKENIZER_MODEL='./T5_Configs/tokenizer.model' 

# Batch size. It must be the same as the one used for the evaluation, since the saved batches have the specified size!!
EVAL_BATCH_SIZE='128'

# Directories that will contain the generated masked files
MASKING_DIR='./masked_dataset'
SAVE_BATCHES_DIR='./masked_dataset/batches'

METRICS_FILENAME='eval_results.json'

# Update jax according to the Conda version
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Generate masked dataset
python ./utils/generate_masked_dataset.py \
       --output_dir="$OUTPUT_DIR" \
       --model_type="t5" \
       --config_name="$CONFIG_DIR" \
       --tokenizer_name="$TOKENIZER_MODEL" \
       --use_fast_tokenizer \
       --validation_file="$DATASET_EVAL" \
       --resume_from_checkpoint="$CHECKPOINT_DIR" \
       --max_seq_length="$MAX_TOKENS_LENGTH" \
       --per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
       --batches_output_dir="$MASKING_DIR" \
       --save_batches_file="$SAVE_BATCHES_DIR"

# List all the checkpoints and generate the predictions
directories=( $(ls -d $OUTPUT_DIR/* | sort --version-sort ) )
for entry in "${directories[@]}"
do
  echo "Generating predictions for checkpoint located in: $entry"
  
  # Launch single prediction on the relative checkpoint against the custom masked dataset created before
  python ./utils/run_t5_mlm_flax.py \
          --output_dir="$OUTPUT_DIR" \
          --model_type="t5" \
          --config_name="$CONFIG_DIR" \
          --tokenizer_name="$TOKENIZER_MODEL" \
          --do_eval \
          --use_fast_tokenizer \
          --train_file="$DATASET_TRAIN" \
          --validation_file="$DATASET_EVAL" \
          --resume_from_checkpoint="$entry" \
          --max_seq_length="$MAX_TOKENS_LENGTH" \
          --per_device_train_batch_size="16" \
          --per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
          --adafactor \
          --learning_rate="0.005" \
          --weight_decay="0.001" \
          --warmup_steps="2000" \
          --overwrite_output_dir \
          --logging_steps="500" \
          --save_steps="5000" \
          --eval_steps="5000" \
          --num_train_epochs="50" \
          --masked_validation_batches="$MASKING_DIR" \
          --metrics_filename="$METRICS_FILENAME"
#         --save_predictions"=$SAVE_BATCHES_DIR" Uncomment to save the predictions generated from the model during evaluation
done

# Get best checkpoint
python ./utils/get_best_checkpoint.py \
    --checkpoints_dir="$OUTPUT_DIR" \
    --metrics_filename="$METRICS_FILENAME" \
    --delta_value="0.005"
