
#!/bin/bash

#SBATCH --job-name=Philosophy-GPT_Wadeva
#SBATCH --time=10:00:00
#SBATCH --mem=4G
#SBATCH --gpus-per-node=3
#SBATCH --output=job-%j.log


module purge
module load tqdm

pip install --upgrade pip
pip install --upgrade wheel
pip install -r ./habrok/requirements-gpu.txt

python3 hyperparams.py
