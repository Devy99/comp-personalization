#!/bin/bash

# This script expects the developer number (in terms of ranking) as parameter.
# If no valid number is provided, the script will exit.
if [ -z "$1" ] || [ "$1" -lt 1 ] || [ "$1" -gt 10 ]; then
    echo "Please provide a valid developer number (1-10) as parameter."
    exit 1
fi

source activate exp1
# # Checks regarding Python, GPU and CUDA
# nvidia-smi
# python --version
# python test.py

# Define dataset variables
SELECTED_DATASETS_DIR="../2-developer-datasets/datasets/selected"
SELECTED_DEVELOPERS_PATH="../2-developer-datasets/results/devs_ranking.txt"
SELECTED_SIZES_PATH="../2-developer-datasets/results/sizes_ranking.txt"

# Define the name of the predictions files
PREDICTIONS_FILENAME_TEST="predictions_test.txt"
PREDICTIONS_FILENAME_EVAL="predictions_eval.txt"

INPUT_COLNAME="masked_method"
TARGET_COLNAME="mask"

# Define the training parameters
MAX_TOKENS_LENGTH="1024"
NUM_EPOCHS="10"
TRAIN_BATCH_SIZE="32"
INFERENCE_BATCH_SIZE="32"

# Define the path for the general models
GENERAL_MODELS_DIR="./general-models"
mkdir $GENERAL_MODELS_DIR

# Define the path of the pre-trained model and the general fine-tuned model
TOKENIZER_MODEL="${GENERAL_MODELS_DIR}/T5_Configs/tokenizer.model"
GENERAL_FINETUNED_MODEL_PATH="${GENERAL_MODELS_DIR}/general-finetuned"

# Define the path of the finetuning, inference and evaluation scripts
FINETUNING_SCRIPT_PATH="../5-baseline-training/fine-tuning/finetuning.py"
EVAL_SCRIPT_PATH="../5-baseline-training/fine-tuning/get_best_checkpoint.py"
INFER_SCRIPT_PATH="../5-baseline-training/fine-tuning/infer.py"

# Move all the selected developers ids to an array
author_ids=()
while IFS= read -r line || [[ -n "$line" ]]
do
        author_ids+=("$line")
done < "$SELECTED_DEVELOPERS_PATH"
id=${author_ids[$1-1]}

# Create the directory to store the developers datasets 
DEVELOPERS_DATASETS_DIR="./dev-datasets"
mkdir $DEVELOPERS_DATASETS_DIR

echo "=================================================="
echo "Training models for author $id - rank $1"
echo "=================================================="

# Generate a CSV file where to store the accuracies for each developer
TRAINING_LOG_FILENAME="training_accuracies_top10_${id}.csv"
rm -f $TRAINING_LOG_FILENAME
echo "author_id,rank,sft1-sft2-all-random" > $TRAINING_LOG_FILENAME

# Initializes the array of the accuracies
author_accuracies=()

# Define paths for the developer
DEVELOPER_DIRPATH="$DEVELOPERS_DATASETS_DIR/developer_$id"
DEVELOPER_DATASETS_DIR="$DEVELOPER_DIRPATH/datasets"
rm -rf $DEVELOPER_DIRPATH
mkdir $DEVELOPER_DIRPATH $DEVELOPER_DATASETS_DIR

# Move the developer dataset to the developer folder
cp -r "$SELECTED_DATASETS_DIR/developer_$id"/* "$DEVELOPER_DATASETS_DIR"

# Define paths for the masked datasets
TRAIN_RANDOM_METHODS_PATH="${DEVELOPER_DATASETS_DIR}/random_changes_train.csv"
EVAL_RANDOM_METHODS_PATH="${DEVELOPER_DATASETS_DIR}/random_changes_eval.csv"
TEST_MASKED_METHODS_PATH="${DEVELOPER_DATASETS_DIR}/developer_masked_methods_test.csv"

# Define the path for fine-tuning the apache dataset with all the developers
SFT1_SFT2_ALL_DIRPATH="$DEVELOPER_DIRPATH/sft1-sft2-all-random"
mkdir $SFT1_SFT2_ALL_DIRPATH

echo "---"
echo "Fine-tuning the random dataset..."
python -u $FINETUNING_SCRIPT_PATH \
    --model_name_or_path="$GENERAL_FINETUNED_MODEL_PATH"  \
    --tokenizer_name="$TOKENIZER_MODEL" \
    --do_train \
    --use_fast_tokenizer \
    --train_file="$TRAIN_RANDOM_METHODS_PATH" \
    --source_column="$INPUT_COLNAME" \
    --target_column="$TARGET_COLNAME" \
    --max_source_length="$MAX_TOKENS_LENGTH" \
    --save_strategy="epoch" \
    --logging_steps="1000" \
    --num_train_epochs="$NUM_EPOCHS"  \
    --save_total_limit="1000" \
    --output_dir="$SFT1_SFT2_ALL_DIRPATH" \
    --per_device_train_batch_size="$TRAIN_BATCH_SIZE" \
    --fp16

# List all the checkpoints and generate the predictions
echo "Generating predictions for all the checkpoints..."
directories=( $(ls -d $SFT1_SFT2_ALL_DIRPATH/*/ | sort --version-sort ) )
for entry in "${directories[@]}"
do
    python -u $INFER_SCRIPT_PATH \
        --model_name_or_path="$entry"  \
        --tokenizer_name="$TOKENIZER_MODEL" \
        --source_column="$INPUT_COLNAME" \
        --target_column="$TARGET_COLNAME" \
        --max_source_length="$MAX_TOKENS_LENGTH" \
        --max_target_length="256" \
        --use_fast_tokenizer \
        --dataset_path="$EVAL_RANDOM_METHODS_PATH" \
        --dataset_split="validation" \
        --batch_size="$INFERENCE_BATCH_SIZE" \
        --output_dir="$entry" \
        --predictions_filename="$PREDICTIONS_FILENAME_EVAL"
done

# Get best checkpoint
echo "Getting the best checkpoint..."
python -u $EVAL_SCRIPT_PATH \
    --eval_file="$EVAL_RANDOM_METHODS_PATH"  \
    --checkpoints_dir="$SFT1_SFT2_ALL_DIRPATH" \
    --predictions_filename="$PREDICTIONS_FILENAME_EVAL"  \
    --target_colname="$TARGET_COLNAME" \
    --checkpoint_filename="$SFT1_SFT2_ALL_DIRPATH/best_checkpoint.txt"

# Extract the path from the best_checkpoint.txt file
BEST_CHECKPOINT_PATH=$(head -n 1 "$SFT1_SFT2_ALL_DIRPATH/best_checkpoint.txt")

echo "Generating predictions for the best checkpoint on the test set..."
python -u $INFER_SCRIPT_PATH \
    --model_name_or_path="$BEST_CHECKPOINT_PATH"  \
    --tokenizer_name="$TOKENIZER_MODEL" \
    --source_column="$INPUT_COLNAME" \
    --target_column="$TARGET_COLNAME" \
    --max_source_length="$MAX_TOKENS_LENGTH" \
    --max_target_length="256" \
    --use_fast_tokenizer \
    --dataset_path="$TEST_MASKED_METHODS_PATH" \
    --dataset_split="test" \
    --batch_size="$INFERENCE_BATCH_SIZE" \
    --output_dir="$SFT1_SFT2_ALL_DIRPATH" \
    --predictions_filename="$PREDICTIONS_FILENAME_TEST"  \
    --save_accuracy_filename="$SFT1_SFT2_ALL_DIRPATH/accuracy_test.txt"

# Extract the saved accuracy
ACCURACY=$(head -n 1 "$SFT1_SFT2_ALL_DIRPATH/accuracy_test.txt")
author_accuracies+=("$ACCURACY")
echo "---"

# Update the CSV file withe the retrieved accuracies
echo "$id,$1,${author_accuracies[0]}" >> $TRAINING_LOG_FILENAME

echo "=================================================="