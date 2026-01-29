#!/bin/bash
#SBATCH --job-name="basic-gumbel"
#SBATCH --mail-user="YOUR_EMAIL"
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --partition=ai
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_output/YOUR_OUTPUT_PATH/output_%A_%a.log
#SBATCH --propagate=NONE
#SBATCH --array=8,11

source activate tradeoff

cd YOUR_PROJECT_PATH

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HUGGING_FACE_HUB_TOKEN=YOUR_ACCESS_TOKEN

# Define parameter arrays
methods=("basic" "gumbel")
models=("llama" "gemma")
ns=(2 3 4)

# Sizes
num_methods=${#methods[@]}
num_models=${#models[@]}
num_ns=${#ns[@]}

# Convert SLURM_ARRAY_TASK_ID to 0-based index
task_id=$((SLURM_ARRAY_TASK_ID - 1))

# Compute indices (method varies slowest, n varies fastest)
method_idx=$(( task_id / (num_models * num_ns) ))
model_idx=$(( (task_id / num_ns) % num_models ))
n_idx=$(( task_id % num_ns ))

# Get the actual parameter values
method=${methods[$method_idx]}
model=${models[$model_idx]}
n=${ns[$n_idx]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method: $method"
echo "Model: $model"
echo "N: $n"

python -m my_experiment.main --method "$method" --model "$model" --n "$n"
