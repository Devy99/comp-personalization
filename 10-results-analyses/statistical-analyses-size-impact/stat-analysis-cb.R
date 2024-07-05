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
res=list(model = c(), p.value = c(), d = c())

# Models to compare: sft1_sft2_dev vs organization small. sft1_sft2_dev is the baseline
model = 'organization_subset'
res$model=c(res$model,as.character(model))
cb_base_df = read.csv(paste0(datasets_path, 'developer_crystalbleu.csv'))
cb_df = read.csv(paste0(datasets_path, model, '_crystalbleu.csv'))

# Wilcoxon test
p.value=wilcox.test(cb_df$crystalbleu,cb_base_df$crystalbleu,paired=TRUE)$p.value
d=cliff.delta(cb_df$crystalbleu,cb_base_df$crystalbleu,paired=TRUE)$estimate

res$p.value=c(res$p.value,p.value)
res$d=c(res$d,d)

# Models to compare: sft1_sft2_all_dev vs random. sft1_sft2_all_dev is the baseline
model = 'baseline_plus'
res$model=c(res$model,as.character(model))
cb_base_df = read.csv(paste0(datasets_path, 'organization_crystalbleu.csv'))
cb_df = read.csv(paste0(datasets_path, model, '_crystalbleu.csv'))

# Wilcoxon test
p.value=wilcox.test(cb_df$crystalbleu,cb_base_df$crystalbleu,paired=TRUE)$p.value
d=cliff.delta(cb_df$crystalbleu,cb_base_df$crystalbleu,paired=TRUE)$estimate

res$p.value=c(res$p.value,p.value)
res$d=c(res$d,d)

# Export to csv
res=data.frame(res)
write.csv(res, file = output_filename)

