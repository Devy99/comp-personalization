#!/bin/bash

# This script expects the developer number (in terms of ranking) as parameter.
# If no valid number is provided, the script will exit.
if [ -z "$1" ] || [ "$1" -lt 1 ] || [ "$1" -gt 10 ]; then
    echo "Please provide a valid developer number (1-10) as parameter."
    exit 1
fi

# This script expects the model type as second parameter.
if [ -z "$2" ] || [ "$2" != "dev" ] && [ "$2" != "org" ]; then
    echo "Please provide 'dev' or 'org' as second argument."
    exit 1
fi

# Define dataset variables
if [ "$2" == "dev" ]; then
    DEVELOPER_EVAL_FILE="./datasets_parsed/developer_${1}/developer_masked_methods_eval.json"
else
    DEVELOPER_EVAL_FILE="./datasets_parsed/developer_${1}/apache_dataset_total_eval.json"
fi

DEVELOPER_TEST_FILE="./datasets_parsed/developer_${1}/developer_masked_methods_test.json"
SELECTED_MODEL_DIR="./${2}/checkpoints-codellama-7b-lora-dev-${1}"

HF_TOKEN="<your_token>"
BASE_MODEL="meta-llama/CodeLlama-7b-hf"

# Define the name of the predictions files
PREDICTIONS_FILENAME_TEST="predictions_test.txt"
PREDICTIONS_FILENAME_EVAL="eval_predictions.json"

# Define the training parameters
MAX_SOURCE_LENGTH="2304"
MAX_TARGET_LENGTH="256"
INFERENCE_BATCH_SIZE="16"

# Define the path of  inference and evaluation scripts
EVAL_SCRIPT_PATH="get_best_checkpoint.py"
INFER_SCRIPT_PATH="inference.py"

# Results directory
RESULTS_DIR="./evaluation/developer_${1}"
mkdir -p $RESULTS_DIR

echo "=========================="
echo " Evaluating developer $1  "
echo "=========================="

# Evaluate the base model on the test set
python3 $INFER_SCRIPT_PATH  --model_name_or_path $BASE_MODEL \
                            --tokenizer_name $BASE_MODEL \
                            --auth_token $HF_TOKEN \
                            --dataset_path $DEVELOPER_TEST_FILE \
                            --max_source_length $MAX_SOURCE_LENGTH \
                            --max_target_length $MAX_TARGET_LENGTH \
                            --batch_size $INFERENCE_BATCH_SIZE \
                            --label "baseline" \
                            --output_dir $RESULTS_DIR 

# List all the checkpoints and generate the predictions
echo "Generating predictions for all the checkpoints..."
directories=( $(ls -d $SELECTED_MODEL_DIR/*/ | sort --version-sort ) )
for entry in "${directories[@]}"
do
    # Evaluate the base model on the test set
    python3 $INFER_SCRIPT_PATH  --adapter_path $entry \
                                --tokenizer_name $BASE_MODEL \
                                --auth_token $HF_TOKEN \
                                --dataset_path $DEVELOPER_EVAL_FILE \
                                --max_source_length $MAX_SOURCE_LENGTH \
                                --max_target_length $MAX_TARGET_LENGTH \
                                --batch_size $INFERENCE_BATCH_SIZE \
                                --label "eval" \
                                --output_dir $entry 
done

# Get best checkpoint
echo "Getting the best checkpoint..."
python -u $EVAL_SCRIPT_PATH --checkpoints_dir="$SELECTED_MODEL_DIR" \
                            --predictions_filename="$PREDICTIONS_FILENAME_EVAL"  \
                            --checkpoint_filename="$SELECTED_MODEL_DIR/best_checkpoint.txt" 

# Extract the path from the best_checkpoint.txt file
BEST_CHECKPOINT_PATH=$(head -n 1 "$SELECTED_MODEL_DIR/best_checkpoint.txt")

echo "Generating predictions for the best checkpoint on the test set..."
# Evaluate the base model on the test set
python3 -u $INFER_SCRIPT_PATH  --adapter_path $BEST_CHECKPOINT_PATH \
                               --tokenizer_name $BASE_MODEL \
                               --auth_token $HF_TOKEN \
                               --dataset_path $DEVELOPER_TEST_FILE \
                               --max_source_length $MAX_SOURCE_LENGTH \
                               --max_target_length $MAX_TARGET_LENGTH \
                               --batch_size $INFERENCE_BATCH_SIZE \
                               --label "best_${2}_finetuned" \
                               --output_dir $RESULTS_DIR 