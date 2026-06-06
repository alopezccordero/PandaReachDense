#!/bin/bash
#SBATCH --job-name=A2Cexperiment
#SBATCH --output=PandaReachDense.out.%j
#SBATCH --partition=gpua30q
#SBATCH --time=12:00:00
#SBATCH --nodes=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating panda-env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate panda-env

echo "Python"
which python
python --version

echo "Start PandaReachDense training"
python3 ~/PandaReachDense/pandareachdense.py

echo "Deactivating environment"
deactivate
