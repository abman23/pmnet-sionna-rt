#!/bin/bash
#SBATCH --account=jl_615_1279
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ylu62702@usc.edu
module purge
eval "$(conda shell.bash hook)"
conda activate autobs

#nvidia-smi

export PYTHONPATH=$PWD:$PYTHONPATH
python dataset_builder/generate_pmap.py