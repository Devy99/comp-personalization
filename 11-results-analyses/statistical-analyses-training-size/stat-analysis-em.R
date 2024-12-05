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

print('EXACT MATCH')

# Init the list of results
res=list(dev = c(), model = c(), p.value = c(), OR = c())

devs=c('developer_1', 'developer_2', 'developer_3', 'developer_4', 'developer_5', 'developer_6', 'developer_7', 'developer_8', 'developer_9', 'developer_10')

# Loop over each developer
for (dev in devs) {
    dev_n=as.numeric(gsub("developer_", "", dev))

    # McNemar test - baseline plus vs organization
    model_col=read.csv(paste0(datasets_path, dev, '/sft1-sft2-all-dev', '/predictions_test.csv'))$correct
    baseline_col=read.csv(paste0(datasets_path, dev, '/rnd/predictions_test.csv'))$correct
    
    mn=mcnemar.exact(baseline_col,model_col)
    p.value=mn$p.value
    or=mn$estimate

    res$dev=c(res$dev,dev_n)
    res$model=c(res$model,'baseline-plus')
    res$p.value=c(res$p.value,p.value)
    res$OR=c(res$OR,or)

    # McNemar test - organization subset vs developer
    model_col=read.csv(paste0(datasets_path, dev, '/sft1-sft2-dev', '/predictions_test.csv'))$correct
    baseline_col=read.csv(paste0(datasets_path, dev,  '/sft1-sft2-all-small', '/predictions_test.csv'))$correct
    
    mn=mcnemar.exact(baseline_col,model_col)
    p.value=mn$p.value
    or=mn$estimate

    res$dev=c(res$dev,dev_n)
    res$model=c(res$model, 'organization-subset')
    res$p.value=c(res$p.value,p.value)
    res$OR=c(res$OR,or)
}

# Generate the dataframes and export to csv
res=data.frame(res)
write.csv(res, file = output_filename)
