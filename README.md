# [Replication Package] Why Personalizing Deep Learning-Based Code Completion Tools Matters

## Introduction
This repository contains the scripts required to reproduce the results described in *"Why Personalizing Deep Learning-Based Code Completion Tools Matters"*. 
In this work, we conduct an empirical study to understand how far it is possible to personalize code completion recommender systems on two levels of granularity: organization-specific and developer-specific.

## Contents 
1. [Prerequisites](#prerequisites)
2. [Datasets and materials](#datasets-and-materials)
3. [Replication of the results](#replication-of-the-results)

## Prerequisites
Below is the list of the languages used for writing the scripts of this repository and the versions used for their execution:
- **Python** (tested on version 3.9.6)
- **R** (tested on version 4.3.2)
### Dependencies
Before running the scripts, you need to download some Python libraries. 
To avoid conflicts with packages installed locally on the machine, create a virtual environment as follows:
  ```sh
   python3 -m venv venv
   ```
and activate it:
  ```sh
   # on Linux / MacOS
   source venv/bin/activate

   # on Windows
   venv\Scripts\activate
   ```

Then, you can install all the required dependencies with the following command:  
   ```sh
   pip3 install -r requirements.txt
   ```
### Hardware 
We ran our experiments using a cluster consisting of 32 NVIDIA A40 and 8 NVIDIA A100 GPUs. You can replicate our work with different GPUs, setting the training configurations according to their capabilities  (see [Training](#training)).

To correctly reproduce the scripts for the model training, you need to install the CUDA toolkit version 12.2.0.

## Datasets and materials
You can find the datasets and the results of our experiments on our [Zenodo repository](https://doi.org/10.5281/zenodo.10817220).

Below, we describe the files stored in the repository. Each directory is divided in "apache" and "spring" to differentiate the datasets of the two organizations.
- **datasets**: contains the datasets used for pre-training and fine-tuning our baseline (folder *general_datasets*) and for the second fine-tuning of developer- and organization-specific models (folder *developer_datasets*). 

    The datasets of the developers are organized as follows:
    - Train / validation / test set containing **developer code changes** ( files developer_masked_methods_\*.csv )
    - Train and validation sets used to train and validate the **organization-specific** model for that particular developer ( files apache_dataset_total_\*.csv and spring-projects_dataset_total_\*.csv )
    - Only for the top 10 developers, train and validation sets used to train and validate the **organization subset** (files apache_dataset_small_\*.csv and spring-projects_dataset_small_*.csv) and the **baseline+** (files random_changes_\*.csv) models.

    Under the directory *developer_datasets*, you can find also the *raw-datasets* folder, which contains the formatted version of the datasets above, used to train the Code Llama models.
- **results**: contains the outcome of our experiments, namely the performance of the developer- and organization-specific models on the developers' test sets using T5 small, T5 large and Codellama, and the analysis of the impact of the training size. 

    In particular, we find the following folders:
    - **predictions**, which contains, for each developer, the predictions of the baseline model (folder *pt-sft1*) and of the developer- (folder *sft1-sft2-dev*) and organization-specific (folder *sft1-sft2-all-dev*) models on the developer test set. For the top 10 developers, you can also find the predictions of the organization subset model (folder *sft1-sft2-all-small*) and the baseline+ model (folder *rnd*). In case of Code Llama results, you can also find the predictions of the baseline fine-tuned on 10k instances under the directory *pt-sft1-ft*.
    - **accuracies**, containing CSV files reporting the performance of each model on the developers' test sets.
    - **crystalBLEU**, containing the CrystalBLEU score on the predictions of each model and for each developer (file crystalbleu.csv).
    - **statistical_analysis**, containing the p-value and the Odds Ratio calculated on each model result (file pvalue_or_em_\*.csv) and the p-value and effect size on the CrystalBLEU distribution (file pvalue_es_cb_\*.csv).

## Replication of the results
In this repository, you can find a list of numbered folders containing scripts for reproducing the results described in our work. To correctly execute the code, it is necessary to run the scripts step-by-step, starting from the first folder (*1-github-miner*). Each directory contains Bash scripts that aid the execution of the Python scripts following the correct flow. Note that with these scripts you can reproduce the results for the Apache organization. However, you can easily adapt them to the organization of your interest by replacing the organization name in the scripts. Also, some scripts require high computational resources to be correctly executed. Hence, we recommend running them on a high-performance machine.

### Datasets generation
The first step is creating the datasets for training our models. 

In the *1-github-miner* folder, we collect commits from developers of the Apache organization, targeting only Java repositories (*1.1-collect-commits*). After mining the relevant commits, we must run the notebook *dataset_analysis.ipynb* that remove noise from our dataset and apply the [gambit](https://github.com/gotec/gambit?tab=readme-ov-file) disambiguation tool to the authors' names and emails.

Then, in the 2-developer-datasets we proceed to get the first 1000 developers by the number of added lines and validate their aliases, extract methods containing developers' code changes from the collected commits, mask the affected lines / blocks and select the top 100 developers by the number of masked instances.

After creating the datasets of the developers, we can generate the pre-training and fine-tuning dataset for our baseline in folder *3-general-dataset*. We collect Java methods from the [codeparrot](https://huggingface.co/datasets/codeparrot/github-code) dataset, pre-process data, remove Apache methods, and mask the collected methods according to the distribution in the developer's dataset. 

Finally, we collect commits from repositories different from Apache and from those used in the training of the baseline (*1-github-miner/1.2-random-commits*) and generate the baseline+ dataset (*4-random-dataset*).

### Training
After producing the datasets, we can start pre-training and fine-tuning the baseline with [T5 small v1.1](https://huggingface.co/google/t5-v1_1-small) (*5-baseline-training*) and [T5 large v1.1](https://huggingface.co/google/t5-v1_1-large) (*6-baseline-training-large*). During the training of the models, we store checkpoints in a dedicated folder. We successively evaluate the best checkpoint on the validation set and pick it for the second fine-tuning step.

In *7-developer-training*, *8-size-impact-training*, and *9-large-training* folders, we reproduce the three experiments performing a second fine-tuning step on top of our base model. Before starting the training script, you must create the *general-models* directory, containing the selected checkpoint for the base model and the model tokenizer. Below is an example for the *7-developer-training* folder:
  ```sh
   # Move to the desired training folder
   cd 7-developer-training

   # Create the general-models directory
   mkdir general-models

   # Copy the base model checkpoint and the tokenizer folder
   cp -r ../5-baseline-training/fine-tuning/T5_Config ./general-models
   cp -r ../5-baseline-training/fine-tuning/checkpoints/checkpoint-10000 ./general-models/general-finetuned
   ```

We trained T5 small models with a batch size of 32 and T5 large models with a batch size of 4. Reproducing our work with different GPUs could require modifying this value to the most appropriate for the used machine. You can update this value by changing the variable "TRAIN_BATCH_SIZE" in the training scripts.

In *10-codellama-training*, instead, we fine-tune the Code Llama model on the developer/organization datasets using LoRA. For dependency compatibility, we recommend to use a different virtual environment with the dependencies specified in the *requirements_peft.txt* file.

### Analysis of the results
Finally, the 11-results-analyses folder contains the code used for analyzing the outcome of the experiments. Starting from the datasets described above, you can reproduce the CrystalBLEU distribution and the statistical analyses reported in the paper. 

Below is an example of how to perform statistical tests and calculate CrystalBLEU distribution for T5 small and T5 large experiments: 

  ```sh
   # Move to the desired folder
   cd 11-results-analyses/statistical-analyses

   # Copy models' predictions of T5 small and large experiments
   cp -r ./<YOUR_PATH>/results/apache/predictions/predictions_top_100_t5small .
   cp -r ./<YOUR_PATH>/results/apache/predictions/predictions_top_10_large .

   # Copy fine-tuning dataset for computing trivially shared n-grams
   cp ./<YOUR_PATH>/datasets/general_datasets/apache/formatted_methods.csv .

   # Run script
   bash run_stats.sh
   ```