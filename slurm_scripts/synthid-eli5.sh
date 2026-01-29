#!/bin/bash
#SBATCH --job-name="synthid_array"
#SBATCH --mail-user="YOUR_EMAIL"
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --partition=ai
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL
#SBATCH --array=0-5
#SBATCH --output=slurm_output/YOUR_OUTPUT_PATH/output_%A_%a.log
#SBATCH --propagate=NONE

source activate tradeoff

cd YOUR_PROJECT_PATH

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HUGGING_FACE_HUB_TOKEN=YOUR_ACCESS_TOKEN
# Define model pairs: (ref_model, model)
MODELS=("google/gemma-7b" "google/gemma-7b" "google/gemma-7b" "huggyllama/llama-7b" "huggyllama/llama-7b" "huggyllama/llama-7b")
REF_MODELS=("google/gemma-2b" "google/gemma-2b" "google/gemma-2b" "JackFram/llama-68m" "JackFram/llama-68m" "JackFram/llama-68m")

# Define n values
NS=(2 3 4 2 3 4)

# Get the model pair and n value for this task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
REF_MODEL=${REF_MODELS[$SLURM_ARRAY_TASK_ID]}
N=${NS[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with model=$MODEL, ref_model=$REF_MODEL, n=$N"

# Execute the Python script with parameters
python -m my_experiment.synthid_experiment_mc \
    --private_key 0 \
    --mc_private_key 1 \
    --model "$MODEL" \
    --ref_model "$REF_MODEL" \
    --method mc_2keys \
    --task eli5 \
    --temperature 1 \
    --top_k 100 \
    --n $N \
    --ds_begain 0 \
    --ds_end 1000 \
    --seed 42 \
    --context_size 4 \
    --folder_name Synthid-data
