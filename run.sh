#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
###SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
##SBATCH --mem=512m
###SBATCH --mem=160G
#SBATCH --time=8:00:00

module load python
module load texlive/2021
source env/bin/activate

# Train the model
srun -u python -u model.py
#srun -u python -u script_mnist.py

