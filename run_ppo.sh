#!/bin/bash
#SBATCH --account=jl_615_1279
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=v100:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ylu62702@usc.edu
module purge
eval "$(conda shell.bash hook)"
conda activate autobs

export PYTHONPATH=$PWD:$PYTHONPATH
python runner/runner_ppo.py