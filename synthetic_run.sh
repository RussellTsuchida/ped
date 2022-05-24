#!/bin/bash -l

###SBATCH --nodes=1
###SBATCH --tasks-per-node=1
###SBATCH --cpus-per-task=5
###SBATCH --gpus-per-node=1
###SBATCH --mem=100GB
###SBATCH --mem=160G

#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

module load python
module load texlive/2021
source env/bin/activate

# Train the model
srun -u python -u script_synthetic.py $1

