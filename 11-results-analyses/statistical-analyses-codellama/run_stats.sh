#!/bin/bash

# Path of the directory containing the predictions
LARGE_PREDICTIONS=./predictions_top_10_codellama/

# Output directories for crystalBLEU
CRYSTALBLEU_DIR_LARGE=./cb_large/

# Models to be compared
MODELS=sft1-sft2-all-dev,sft1-sft2-dev

# Dataset used to find trivially shared n-grams
JAVA_DATASET=./formatted_methods.csv

echo "===================================================================================================="
echo "Computing p-value and OR for the large models and p-value and effect size for crystalBLEU - large models"
echo "===================================================================================================="


# P-value and OR
Rscript stat-analysis-em.R $LARGE_PREDICTIONS $MODELS pvalue_or_em_codellama.csv

# CrystalBLEU
python3 -u comp_crystalbleu.py \
    --predictions_dir $LARGE_PREDICTIONS \
    --training_dataset_path $JAVA_DATASET \
    --results_dir $CRYSTALBLEU_DIR_LARGE 

# P-value and effect size
Rscript stat-analysis-cb.R $CRYSTALBLEU_DIR_LARGE $MODELS pvalue_es_cb_codellama.csv