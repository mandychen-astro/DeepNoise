#!/bin/bash

#SBATCH --job-name=PKS0405-123_OB1EXP1
#SBATCH --output=./sbatch_code/PKS0405-123_OB1EXP1.out
#SBATCH --error=./sbatch_code/PKS0405-123_OB1EXP1.err
#SBATCH --time=05:30:00
#SBATCH --partition=broadwl
#SBATCH --account=pi-hwchen
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Add lines here to run your computations.

module load python
conda activate /home/mandychen/conda_envs/pytorch
python PKS0405-123_OB1EXP1_train.py
