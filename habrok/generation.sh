#!/bin/bash

#SBATCH --job-name=GPT_generation
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=generation-%j.log


module purge
module load tqdm

pip install --upgrade pip
pip install --upgrade wheel
pip install -q -r /home1/s4790820/llm/Philosophy-GPT/habrok/requirements-gpu.txt

python3 run.py
