# Clear the environment
rm(list=ls())

# Download the required libraries if not installed
if (!require("exact2x2")) install.packages("exact2x2")
if (!require("xtable")) install.packages("xtable")
if (!require("effsize")) install.packages("effsize")
if (!require("nortest")) install.packages("nortest")

# Load the required libraries
library(exact2x2)
library(xtable)
library(effsize)
library(nortest)

# Take args from command line
args <- commandArgs(trailingOnly=TRUE)
datasets_path <- args[1]
output_filename <- args[2]

print('CRYSTAL BLEU')

# Init the list of results
res=list(dev = c(), model = c(), p.value = c(), d = c())
devs=list.dirs(path = datasets_path, full.names = FALSE, recursive = FALSE)

for (dev in devs) {
    dev_n=as.numeric(gsub("developer_", "", dev))

    # Models to compare: Developer vs Organization subset.
    cb_df=read.csv(paste0(datasets_path, dev, '/sft1-sft2-all-small/crystalbleu.csv'))
    cb_df$exact_match <- ifelse(cb_df$correct == "True",1,0)
    cb_base_df=read.csv(paste0(datasets_path, dev, '/sft1-sft2-dev/crystalbleu.csv'))
    cb_base_df$exact_match <- ifelse(cb_base_df$correct == "True",1,0)

    # Remove rows where both models are correct
    merged_df = data.frame(cb_df$crystalbleu, cb_base_df$crystalbleu, cb_base_df$exact_match, cb_df$exact_match)
    colnames(merged_df) = c('crystalbleu_target', 'crystalbleu_baseline', 'correct_baseline', 'correct_target')
    merged_df$sum = merged_df$correct_baseline + merged_df$correct_target
    merged_df = merged_df[merged_df$sum != 2,]

    # Wilcoxon test
    p.value=wilcox.test(merged_df$crystalbleu_target,merged_df$crystalbleu_baseline,paired=TRUE)$p.value
    d=cliff.delta(merged_df$crystalbleu_target,merged_df$crystalbleu_baseline,paired=TRUE)$estimate

    model = 'organization-subset'
    res$dev=c(res$dev,dev_n)
    res$model=c(res$model,as.character(model))
    res$p.value=c(res$p.value,p.value)
    res$d=c(res$d,d)

    # Models to compare: Organization vs Baseline+.
    cb_df=read.csv(paste0(datasets_path, dev, '/rnd/crystalbleu.csv'))
    cb_df$exact_match <- ifelse(cb_df$correct == "True",1,0)
    cb_base_df=read.csv(paste0(datasets_path, dev, '/sft1-sft2-all-dev/crystalbleu.csv'))
    cb_base_df$exact_match <- ifelse(cb_base_df$correct == "True",1,0)
    
    # Remove rows where both models are correct
    merged_df = data.frame(cb_df$crystalbleu, cb_base_df$crystalbleu, cb_base_df$exact_match, cb_df$exact_match)
    colnames(merged_df) = c('crystalbleu_target', 'crystalbleu_baseline', 'correct_baseline', 'correct_target')
    merged_df$sum = merged_df$correct_baseline + merged_df$correct_target
    merged_df = merged_df[merged_df$sum != 2,]

    # Wilcoxon test
    p.value=wilcox.test(merged_df$crystalbleu_target,merged_df$crystalbleu_baseline,paired=TRUE)$p.value
    d=cliff.delta(merged_df$crystalbleu_target,merged_df$crystalbleu_baseline,paired=TRUE)$estimate

    model = 'baseline-plus'
    res$dev=c(res$dev,dev_n)
    res$model=c(res$model,as.character(model))
    res$p.value=c(res$p.value,p.value)
    res$d=c(res$d,d)
}

# Export to csv
res=data.frame(res)
write.csv(res, file = output_filename)

