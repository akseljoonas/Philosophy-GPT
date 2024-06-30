#!/bin/bash

#SBATCH --job-name=GPT_gen
#SBATCH --time=6:00:00
#SBATCH --mem=124G
#SBATCH --gpus-per-node=2
#SBATCH --output=gen-%j.log


module purge
module load tqdm

pip install --upgrade pip
pip install --upgrade wheel
pip install -q -r /home1/s4790820/llm/Philosophy-GPT/habrok/requirements-gpu.txt

python3 run.py
