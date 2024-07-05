#!/bin/bash

# Add your GitHub token as first argument of the script
GITHUB_TOKEN=$1 

# Create directory to store the results
OUTPUT_DIR="./results"
mkdir $OUTPUT_DIR

# Path to the commits dataset and ext data directory
COMMITS_FILENAME="../1-github-miner/1.1-collect-commits/commits_with_authors.csv"
EXT_DATA_DIR='../1-github-miner/1.1-collect-commits/ext-data'

VALID_AUTHORS_FILENAME="./$OUTPUT_DIR/valid_authors.txt"

# Launch script to define valid author ids for the developers with the highest number of added lines of code
echo "Generating valid author ids..."
python -u ./2.1-retrieve-developers/validate_aliases.py \
    --input_filepath="$COMMITS_FILENAME" \
    --output_filepath="$VALID_AUTHORS_FILENAME" \
    --token="$GITHUB_TOKEN" 

# Move all the valid author ids to an array
author_ids=()
while IFS= read -r line || [[ -n "$line" ]]
do
    author_ids+=("$line")
done < "$VALID_AUTHORS_FILENAME"

# Create the directory to store the developers datasets 
DATASETS_DIR="./datasets"
EXTRACTED_DATASETS_PATH="$DATASETS_DIR/extracted"
SELECTED_DATASETS_PATH="$DATASETS_DIR/selected"

mkdir $DATASETS_DIR
mkdir $EXTRACTED_DATASETS_PATH
mkdir $SELECTED_DATASETS_PATH

for id in "${author_ids[@]}"; do
    echo "Retrieving data for author $id"

    # Define paths for the retrieved methods
    RETRIEVED_METHODS_DIRPATH="$EXTRACTED_DATASETS_PATH/developer_$id"
    DATASET_FILEPATH="$RETRIEVED_METHODS_DIRPATH/extracted.csv"

    # Launch script to retrieve the methods modified by the developer
    python -u ./2.2-extract-dataset/retrieve_methods.py \
        --commits_filepath="$COMMITS_FILENAME" \
        --ext_dir="$EXT_DATA_DIR" \
        --author_id="$id" \
        --output_dir="$RETRIEVED_METHODS_DIRPATH"

    echo "Generating token masked dataset for author $id"

    # Define paths for token masked dataset
    MASKED_METHODS_PATH="${RETRIEVED_METHODS_DIRPATH}/developer_masked_methods.csv"
    rm -f "$MASKED_METHODS_PATH"

    # Mask the datasets
    python -u ./2.2-extract-dataset/mask_dataset.py \
        --input_filepath="$DATASET_FILEPATH" \
        --output_filepath="$MASKED_METHODS_PATH"  
done
