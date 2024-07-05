#!/bin/bash

# This script runs the miner of random commits. It collects commits from a set of random repositories.
GHSEARCH_CSV="./ghsearch-result.csv" 
GENERAL_TRAINING_PATH="../3-general-datasets/datasets/formatted_methods.csv"
NUM_REPOS=3000
REPOS_FILEPATH="./random_repos.json" 
COMMITS_FILEPATH="./random_commits.csv" 
EXT_DIRNAME="random-ext-data" 

echo ===================== RETRIEVE RANDOM REPOS =========================

if [ ! -f "$GENERAL_TRAINING_PATH" ]; then
    echo "The file $GENERAL_TRAINING_PATH does not exist. Execute the script 3-general-datasets/run_datasets_generation.sh first."
    exit 1
fi

# Collect commits from the selected repositories
python -u ./extract_random_repos.py \
    --input="$GHSEARCH_CSV" \
    --num_repos="$NUM_REPOS" \
    --output="$REPOS_FILEPATH" \
    --general_training_path="$GENERAL_TRAINING_PATH"

echo =====================================================================

echo ========================= MINE COMMITS ==============================
# Collect commits from the selected repositories
python -u ../1.1-collect-commits/repo_miner.py \
    --repos_filepath="$REPOS_FILEPATH" \
    --commits_filepath="$COMMITS_FILEPATH" \
    --ext_dir="$EXT_DIRNAME" \

echo =====================================================================