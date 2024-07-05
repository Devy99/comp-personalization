#!/bin/bash
#SBATCH --array=1-100
#SBATCH --job-name=training_pipeline_parallel
#SBATCH --output=training_pipeline_parallel_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=1000:00:00

# Load any necessary modules
ml CUDA/12.2.0

# Run the training pipeline script
time ./run_training_pipeline_parallel.sh $SLURM_ARRAY_TASK_ID