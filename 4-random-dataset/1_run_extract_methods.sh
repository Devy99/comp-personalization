#!/bin/bash

COMMIT_FILEPATH="../1-github-miner/1.2-random-commits/random_commits.csv" 
PREPROCESSED_COMMIT_FILEPATH="./clean_random_commits.csv" 

echo =================== PRE-PROCESS COMMITS FILE =========================
echo "Cleaning the commits CSV file..."
python3 -u ./clean_commits.py \
    --input="$COMMIT_FILEPATH" \
    --output="$PREPROCESSED_COMMIT_FILEPATH" 
echo =====================================================================

OUTPUT_DIR="result"
EXT_DIRNAME="../1-github-miner/1.2-random-commits/random-ext-data" 
REMAINING_REPOS_FILEPATH="./remaining_repos.txt"

repos=()
while IFS= read -r line || [[ -n "$line" ]]
do
    repos+=("$line")
done < "$REMAINING_REPOS_FILEPATH"

echo ================== RETRIEVE DEVELOPERS DATA   ========================
echo "Retrieving the developers data..."
for repo in "${repos[@]}"
do
    echo "Retrieving the developers data for the repository: $repo"
    python3 -u ./retrieve_methods.py \
        --commits_filepath="$PREPROCESSED_COMMIT_FILEPATH" \
        --ext_dir="$EXT_DIRNAME" \
        --repository="$repo" \
        --output_dir="$OUTPUT_DIR" 

    NORMALIZED_REPO=$(echo "$repo" | tr / _)

    PRODUCED_FILEPATH="${OUTPUT_DIR}/${NORMALIZED_REPO}_extracted.csv"
    MASKED_FILEPATH="${OUTPUT_DIR}/${NORMALIZED_REPO}_masked.csv"

    echo "Masking the raw code for the repository: $repo"
    python3 -u ./mask_dataset.py \
        --input_filepath="$PRODUCED_FILEPATH" \
        --output_filepath="$MASKED_FILEPATH"
done

echo =====================================================================