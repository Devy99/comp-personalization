#!/bin/bash

# Compute p-value and OR for the small models and p-value and effect size for crystalBLEU - small models
SMALL_PREDICTIONS=./predictions_organization_subset_baselinep/
JAVA_DATASET=./formatted_methods.csv
CRYSTALBLEU_DIR=./cb-organization-subset-baselinep/

# P-value and OR
Rscript stat-analysis-em.R $SMALL_PREDICTIONS pvalue_or_em_org_sub_baselinep.csv

# CrystalBLEU
python3 -u comp_crystalbleu.py \
    --predictions_dir $SMALL_PREDICTIONS \
    --training_dataset_path $JAVA_DATASET \
    --results_dir $CRYSTALBLEU_DIR 

# P-value and effect size
Rscript stat-analysis-cb.R $CRYSTALBLEU_DIR pvalue_es_cb_org_sub_baselinep.csv
