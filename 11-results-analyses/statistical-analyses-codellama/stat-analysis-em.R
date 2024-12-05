# Clear the environment
rm(list=ls())

# Download the required libraries if not installed
if (!require("exact2x2")) install.packages("exact2x2")
if (!require("xtable")) install.packages("xtable")
if (!require("effsize")) install.packages("effsize")
if (!require("nortest")) install.packages("nortest")
if (!require("jsonlite")) install.packages("jsonlite")

# Load the required libraries
library(exact2x2)
library(xtable)
library(effsize)
library(nortest)
library(jsonlite)

# Take args from command line
args <- commandArgs(trailingOnly=TRUE)
datasets_path <- args[1]
models <- args[2]
models <- unlist(strsplit(models, ","))
output_filename <- args[3]

print('EXACT MATCH')

# Init the list of results
res=list(dev = c(), model = c(), p.value = c(), OR = c())

devs=list.dirs(path = datasets_path, full.names = FALSE, recursive = FALSE)

# Loop over each developer
for (dev in devs) {
    dev_n=as.numeric(gsub("developer_", "", dev))

    for (model in models) {
        model_col=read.csv(paste0(datasets_path, dev, '/', model, '/predictions_test.csv'))$exact_match
        baseline_col=read.csv(paste0(datasets_path, dev, '/pt-sft1-ft/predictions_test.csv'))$exact_match
        
        # McNemar test
        mn=mcnemar.exact(baseline_col,model_col)
        p.value=mn$p.value
        or=mn$estimate

        # Change model name
        if (model == 'pt-sft1') {
            model = 'baseline'
        } else if (model == 'best_dev_finetuned') {
            model = 'developer'
        } else if (model == 'best_org_finetuned') {
            model = 'organization'
        }

        res$dev=c(res$dev,dev_n)
        res$model=c(res$model,as.character(model))
        res$p.value=c(res$p.value,p.value)
        res$OR=c(res$OR,or)
    }
}

# Generate the dataframes and export to csv
res=data.frame(res)

# Sort dataframe by dev
res=res[order(res$dev),]
write.csv(res, file = output_filename)
