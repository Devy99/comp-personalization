#!/bin/bash

# Variables for collect_repos.py script
REPOS_ORGANIZATION="apache" # Name of the organization (on GitHub) to mine
LANGUAGES="Java" # Target programming language
REPOS_FILEPATH="./apache_repos.json" # Path of the file where to store the collected repositories 
AUTH_TOKEN="<TOKEN>" # GitHub API token for the usage rate

# Variables for collect_repos.py script
COMMITS_FILEPATH="./commits.csv" # Path of the file where to store the collected commits 
EXT_DIRNAME="ext-data" # Name of the directory where to store the external data of each commit


echo ==================== RETRIEVE REPOSITORIES ==========================
# Launch script to retrieve the target repositories
python collect_repos.py \
    --org="$REPOS_ORGANIZATION" \
    --languages="$LANGUAGES" \
    --output_filepath="$REPOS_FILEPATH" \
    --token="$AUTH_TOKEN" 

echo =====================================================================

echo ========================= MINE COMMITS ==============================
# Collect commits from the selected repositories
python -u repo_miner.py \
    --repos_filepath="$REPOS_FILEPATH" \
    --commits_filepath="$COMMITS_FILEPATH" \
    --ext_dir="$EXT_DIRNAME" \

echo =====================================================================