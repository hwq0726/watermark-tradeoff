#!/bin/bash
#SBATCH --job-name="gumbel-gemma"
#SBATCH --mail-user="YOUR_EMAIL"
#SBATCH --time=05:00:00
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

# Execute the Python script with the selected parameters
python -m my_experiment.main --method gumbel_basic --model gemma --n 1
