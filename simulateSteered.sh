#!/bin/bash
#SBATCH -A research
#SBATCH -c 10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python3 simulateSteered.py 

