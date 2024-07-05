#!/bin/bash

# This script runs the miner of random commits. It collects commits from a set of random repositories.
RESULTS_DIR="./result" 
FINETUNING_PATH="../2-developer-datasets/datasets/finetuning_methods.csv"
EXTRACTED_DATASETS_DIR="../2-developer-datasets/datasets/extracted"
SELECTED_DATASETS_DIR="../2-developer-datasets/datasets/selected"
SELECTED_DEVELOPERS_PATH="../2-developer-datasets/results/selected_developers_ids.txt"

echo =================== GENERATE RANDOM DATASET =========================
# Collect commits from the selected repositories
python3 -u ./generate_random_dataset.py \
    --results_dir="$RESULTS_DIR" \
    --finetuning_path="$FINETUNING_PATH" \
    --extracted_developer_datasets="$EXTRACTED_DATASETS_DIR" \
    --selected_developer_datasets="$SELECTED_DATASETS_DIR" 
echo =====================================================================

# Generate a folder for logging
LOG_DIR="./overlapping_logs"
rm -r $LOG_DIR
mkdir $LOG_DIR

echo ===================== FINE-TUNING vs DEVELOPER DATASETS ==========================
echo "Checking fine-tuning dataset vs developer dataset methods..."

# Move all the selected developers ids to an array
author_ids=()
while IFS= read -r line || [[ -n "$line" ]]
do
        author_ids+=("$line")
done < "$SELECTED_DEVELOPERS_PATH"

# Retrieve developer datasets
for id in "${author_ids[@]}"; do
    echo "Checking dataset for author $id"
    EXTRACTED_DEV_METHODS_PATH="$EXTRACTED_DATASETS_DIR/developer_$id/extracted.csv"
    SELECTED_RANDOM_PATH="$RESULTS_DIR/developer_$id/random_changes_train.csv"

    # Check for same methods
    echo "Overlapping methods: "
    python -u ../../utils/check_overlapping.py \
            --first_input="$SELECTED_RANDOM_PATH" \
            --second_input="$EXTRACTED_DEV_METHODS_PATH" \
            --first_column="method" \
            --second_column="method" \
            --file_type="csv" | tee "${LOG_DIR}/dev_${id}_finetuning_train_vs_dev_methods.txt"
    
    # Check for same masked methods
    RETRIEVED_METHODS_PATH="$SELECTED_DATASETS_DIR/developer_$id/developer_masked_methods_test.csv"
    echo "Overlapping masked methods: "
    python -u ../../utils/check_overlapping.py \
            --first_input="$SELECTED_RANDOM_PATH" \
            --second_input="$RETRIEVED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/dev_${id}_finetuning_train_vs_dev_masked_methods.txt"
done
echo =====================================================================