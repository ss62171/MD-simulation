#!/bin/bash
#SBATCH -A research
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python3 simulateSteered.py 

