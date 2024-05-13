#!/bin/bash

#SBATCH --job-name=PKS0405-123_OB1EXP1
#SBATCH --output=./PKS0405-123_OB1EXP1.out
#SBATCH --error=./PKS0405-123_OB1EXP1.err
#SBATCH --time=05:30:00
#SBATCH --nodes=1        
#SBATCH --partition=kicp-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kicp
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Add lines here to run your computations.

module load python/anaconda-2021.05
module load cuda/11.2
module load cudnn/11.2
nvidia-smi
python PKS0405-123_OB1EXP1_train.py
