#!/bin/bash
#SBATCH --job-name="synthid_basic-llama"
#SBATCH --mail-user="YOUR_EMAIL"
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --partition=ai
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output/YOUR_OUTPUT_PATH/output_%A.log
#SBATCH --propagate=NONE

source activate tradeoff

cd YOUR_PROJECT_PATH

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HUGGING_FACE_HUB_TOKEN=YOUR_ACCESS_TOKEN
# models: google/gemma-2b, google/gemma-7b, facebook/opt-1.3b, facebook/opt-6.7b
# Qwen/Qwen3-8B, Qwen/Qwen3-0.6B
# huggyllama/llama-7b, JackFram/llama-68m
# Execute the Python script with parameters
python -m my_experiment.synthid_experiment_basic \
    --private_key 0 \
    --model "huggyllama/llama-7b" \
    --task oeg \
    --temperature 1 \
    --top_k 100 \
    --n 1 \
    --ds_cut_len 1000 \
    --ds_begain 0 \
    --ds_end 1000 \
    --seed 42 \
    --context_size 4
