#!/bin/bash

#SBATCH --job-name=GPT_Wadeva
#SBATCH --time=16:00:00
#SBATCH --mem=4G
#SBATCH --gpus-per-node=4
#SBATCH --output=job-%j.log


module purge
module load tqdm

pip install --upgrade pip
pip install --upgrade wheel
pip install -r /home1/s4790820/llm/Philosophy-GPT/habrok/requirements-gpu.txt

python3 run.py
