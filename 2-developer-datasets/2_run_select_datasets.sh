#!/bin/bash

# Define variables
OUTPUT_DIR="./results"
EXTRACTED_DATASETS_DIR="./datasets/extracted"
SELECTED_DATASETS_DIR="./datasets/selected"
SELECTED_DEVELOPERS_PATH="./results/selected_developers_ids.txt"

echo "Selecting developers with the highest number of masked methods..."
python -u ./2.3-select-developers/select_developers.py \
    --datasets_dir="$EXTRACTED_DATASETS_DIR" \
    --selected_dir="$SELECTED_DATASETS_DIR" \
    --output_dir="$OUTPUT_DIR" \

echo "Generating train, validation and test sets for the selected developers..."
python -u ./2.3-select-developers/prepare_selected_datasets.py \
    --developers_list_filepath="$SELECTED_DEVELOPERS_PATH" \
    --datasets_dir="$EXTRACTED_DATASETS_DIR" \
    --output_dir="$SELECTED_DATASETS_DIR" \
    --results_dir="$OUTPUT_DIR" 

DEVS_RANKING_PATH="$OUTPUT_DIR/devs_ranking.txt"
echo "Retrieving the distribution of the number of masked tokens per developer..."
python -u ./2.3-select-developers/count_masked_tokens.py \
    --developers_list_filepath="$DEVS_RANKING_PATH" \
    --datasets_dir="$SELECTED_DATASETS_DIR" \
    --results_dir="$OUTPUT_DIR" 

# Move all the selected developers ids to an array
author_ids=()
while IFS= read -r line || [[ -n "$line" ]]
do
        author_ids+=("$line")
done < "$SELECTED_DEVELOPERS_PATH"

# Generate apache datasets
for id in "${author_ids[@]}"; do
    echo "Generating apache data for author $id"

    # Define paths for the retrieved methods
    RETRIEVED_METHODS_DIRPATH="$SELECTED_DATASETS_DIR/developer_$id"
    LOG_DIR="$RETRIEVED_METHODS_DIRPATH/duplicates_log"
    mkdir $LOG_DIR

    # Define paths for the masked dataset
    TRAIN_MASKED_METHODS_PATH="${RETRIEVED_METHODS_DIRPATH}/developer_masked_methods_train.csv"
    EVAL_MASKED_METHODS_PATH="${RETRIEVED_METHODS_DIRPATH}/developer_masked_methods_eval.csv"
    TEST_MASKED_METHODS_PATH="${RETRIEVED_METHODS_DIRPATH}/developer_masked_methods_test.csv"

    # Check for overlapping with the eval and test sets - train dataset
    python -u ../utils/check_overlapping.py \
            --first_input="$TRAIN_MASKED_METHODS_PATH" \
            --second_input="$EVAL_MASKED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/dataset_train_vs_eval_log.txt"
            
    python -u ../utils/check_overlapping.py \
            --first_input="$TRAIN_MASKED_METHODS_PATH" \
            --second_input="$TEST_MASKED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/dataset_train_vs_test_log.txt"

    APACHE_TOTAL_PATH="$RETRIEVED_METHODS_DIRPATH/apache_dataset_total_train.csv"
    APACHE_TOTAL_EVAL_PATH="$RETRIEVED_METHODS_DIRPATH/apache_dataset_total_eval.csv"
    APACHE_SMALL_PATH="$RETRIEVED_METHODS_DIRPATH/apache_dataset_small_train.csv"
    APACHE_SMALL_EVAL_PATH="$RETRIEVED_METHODS_DIRPATH/apache_dataset_small_eval.csv"

    echo "Generating apache dataset for the masked methods"

    # Generate the apache dataset containing all developer block-masked methods with / without the developer
    python -u ./2.3-select-developers/generate_apache_datasets.py \
        --datasets_dir="$SELECTED_DATASETS_DIR" \
        --reference_dataset="$TRAIN_MASKED_METHODS_PATH" \
        --author_id="$id" \
        --output_dir="$RETRIEVED_METHODS_DIRPATH" \
        --datasets_filename="developer_masked_methods.csv" \
        --datasets_to_compare $EVAL_MASKED_METHODS_PATH $TEST_MASKED_METHODS_PATH 

    # Check for overlapping with the eval and test sets
    python -u ../utils/check_overlapping.py \
            --first_input="$APACHE_TOTAL_PATH" \
            --second_input="$APACHE_TOTAL_EVAL_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/apache_total_vs_eval_log.txt"
            
    python -u ../utils/check_overlapping.py \
            --first_input="$APACHE_TOTAL_PATH" \
            --second_input="$TEST_MASKED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/apache_total_vs_test_log.txt"


    python -u ../utils/check_overlapping.py \
            --first_input="$APACHE_SMALL_PATH" \
            --second_input="$APACHE_SMALL_EVAL_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/apache_small_vs_eval_log.txt"
            
    python -u ../utils/check_overlapping.py \
            --first_input="$APACHE_SMALL_PATH" \
            --second_input="$TEST_MASKED_METHODS_PATH" \
            --first_column="masked_method" \
            --first_target="mask" \
            --second_column="masked_method" \
            --second_target="mask" \
            --file_type="csv" | tee "${LOG_DIR}/apache_small_vs_test_log.txt"
done

# Check if the apache datasets are valid
python -u ./2.3-select-developers/check_datasets.py \
    --datasets_dir="$SELECTED_DATASETS_DIR"