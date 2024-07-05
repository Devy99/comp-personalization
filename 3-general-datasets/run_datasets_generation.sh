#!/bin/bash

echo ===================== EXTRACT RAW METHODS ===========================
# File containing the raw Java methods extracted and filtered from the complete dataset
RAW_DATASET="./datasets/raw_methods.csv"

# Launch script to retrieve and filter raw Java methods
python -u ./3.1-raw-dataset/extract_dataset.py \
    --output="$RAW_DATASET" 

echo =====================================================================


echo ===================== PROCESS RAW METHODS ===========================
# Variables for prepare_dataset.py script
RAW_DATASET_DIR="./datasets/dataset_raw"
FORMATTED_DATASET="./datasets/formatted_methods.csv" 
GHS_SELECTED_REPOS="./datasets/ghsearch-result.csv" 

# Launch script to process the raw Java methods
python -u ./3.2-processed-dataset/prepare_dataset.py \
    --input="$RAW_DATASET_DIR" \
    --output="$FORMATTED_DATASET" \
    --filter="$GHS_SELECTED_REPOS"

echo =====================================================================


echo ==================== DATASET SPLITTING ==========================
# Splitted datasets
PRETRAINING_DATASET="./datasets/pretraining_methods.csv"
FINETUNING_DATASET="./datasets/finetuning_methods.csv" 

# Split dataset in pre-training [40%] and fine-tuning [60%] by repository
python -u ../utils/split_dataset.py \
        --input="$FORMATTED_DATASET" \
        --split_ratio="40" \
        --first_split_name="$PRETRAINING_DATASET" \
        --second_split_name="$FINETUNING_DATASET" \
        --split_by="repository" \
        --file_type="csv"

echo =====================================================================


echo ==================== PRE-TRAINING SPLITTING ==========================
# Generate pre-training folder
mkdir ./datasets/pretraining/

# Splitted pre-training TXT datasets
PRETRAINING_TRAIN="./datasets/pretraining/pretraining_train.txt"  
PRETRAINING_EVAL="./datasets/pretraining/pretraining_eval.txt"

# Split pre-training dataset in training set [90%] and validation set [10%]
python -u ../utils/split_dataset.py \
        --input="$PRETRAINING_DATASET" \
        --split_ratio="90" \
        --first_split_name="$PRETRAINING_TRAIN" \
        --second_split_name="$PRETRAINING_EVAL" \
        --split_by="repository" \
        --retain_only="formatted" \
        --file_type="csv"

echo =====================================================================

# Do a backup copy of the fine-tuning dataset
cp "$FINETUNING_DATASET" "${FINETUNING_DATASET}_backup.csv"

echo ===================== FINE-TUNING CLEANING ==========================
# Clean fine-tuning dataset
echo "Cleaning fine-tuning dataset from apache dataset methods..."

CLEAN_FINETUNING_DATASET="./datasets/clean_finetuning_methods.csv" 
EXTRACTED_DATASETS_DIR="../2-developer-datasets/datasets/extracted"

python -u ./3.3-finetuning-dataset/remove_apache_instances.py \
        --finetuning_path="$FINETUNING_DATASET" \
        --datasets_dir="$EXTRACTED_DATASETS_DIR" \
        --datasets_filename="extracted.csv" \
        --output_path="$CLEAN_FINETUNING_DATASET" 
echo =====================================================================

echo ==================== FINE-TUNING SPLITTING ==========================
# Generate fine-tuning folder
mkdir ./datasets/finetuning/

# Mask the fine-tuning dataset
FINETUNING_METHODS="./datasets/finetuning/finetuning_masked_methods.csv" 
DEVELOPERS_DATASET_PATH="../2-developer-datasets/results/devs_distribution.csv" 

python -u ./3.3-finetuning-dataset/mask_dataset.py \
        --input_filepath="$CLEAN_FINETUNING_DATASET" \
        --output_filepath="$FINETUNING_METHODS" \
        --devs_total_filepath="$DEVELOPERS_DATASET_PATH"

# Splitted fine-tuning methods
FINETUNING_METHODS_TRAIN="./datasets/finetuning/finetuning_methods_train.csv" 
FINETUNING_METHODS_EVAL="./datasets/finetuning/finetuning_methods_eval.csv"

# Split dataset in training set [90%] and validation set [10%]
python -u ../utils/split_dataset.py \
        --input="$FINETUNING_METHODS" \
        --split_ratio="90" \
        --first_split_name="$FINETUNING_METHODS_TRAIN" \
        --second_split_name="$FINETUNING_METHODS_EVAL" \
        --split_by="repository" \
        --file_type="csv"

# Remove from the validation set overlapping instances with the train set
python -u ../utils/remove_duplicates.py \
        --subject_dataset="$FINETUNING_METHODS_EVAL" \
        --ref_dataset="$FINETUNING_METHODS_TRAIN" \
        --columns_to_check "method"

python -u ../utils/remove_duplicates.py \
        --subject_dataset="$FINETUNING_METHODS_EVAL" \
        --ref_dataset="$FINETUNING_METHODS_TRAIN" \
        --columns_to_check "masked_method" "mask"

echo =====================================================================

# Generate a folder for logging
LOG_DIR="./overlapping_logs"
rm -r $LOG_DIR
mkdir $LOG_DIR

# Check overlapping among fine-tuning dataset splits for masking
echo ============ CHECK FINE-TUNING TRAIN vs EVAL  =================

# Check for same methods
python -u ../utils/check_overlapping.py \
        --first_input="$FINETUNING_METHODS_TRAIN" \
        --second_input="$FINETUNING_METHODS_EVAL" \
        --first_column="method" \
        --second_column="method" \
        --file_type="csv" | tee "${LOG_DIR}/finetuning_train_vs_eval_masked_overlapping_methods.txt"

# Check for same masked methods
python -u ../utils/check_overlapping.py \
        --first_input="$FINETUNING_METHODS_TRAIN" \
        --second_input="$FINETUNING_METHODS_EVAL" \
        --first_column="masked_method" \
        --first_target="mask" \
        --second_column="masked_method" \
        --second_target="mask" \
        --file_type="csv" | tee "${LOG_DIR}/finetuning_train_vs_eval_masked.txt"

echo =====================================================================

echo ===================== FINE-TUNING vs DEVELOPER DATASETS ==========================
# Clean fine-tuning dataset
echo "Checking fine-tuning dataset vs developer dataset methods..."

EXTRACTED_DATASETS_DIR="../2-developer-datasets/datasets/extracted"
SELECTED_DATASETS_DIR="../2-developer-datasets/datasets/selected"
SELECTED_DEVELOPERS_PATH="../2-developer-datasets/results/selected_developers_ids.txt"

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

    # Check for same methods
    echo "Overlapping methods: "
    python -u ../utils/check_overlapping.py \
            --first_input="$FINETUNING_METHODS_TRAIN" \
            --second_input="$EXTRACTED_DEV_METHODS_PATH" \
            --first_column="method" \
            --second_column="method" \
            --file_type="csv" | tee "${LOG_DIR}/dev_${id}_finetuning_train_vs_dev_methods.txt"
    
    # Check for same masked methods
    RETRIEVED_METHODS_PATH="$SELECTED_DATASETS_DIR/developer_$id/developer_masked_methods_test.csv"
    echo "Overlapping masked methods: "
    python -u ../utils/check_overlapping.py \
            --first_input="$FINETUNING_METHODS_TRAIN" \
            --second_input="$RETRIEVED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/dev_${id}_finetuning_train_vs_dev_masked_methods.txt"
done
echo =====================================================================