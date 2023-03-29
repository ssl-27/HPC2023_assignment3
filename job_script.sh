#!/bin/bash

#SBATCH -A training2302
#SBATCH --partition=dc-cpu
#SBATCH --job-name=TEST
##SBATCH --nodes=10
##SBATCH --ntasks-per-node=4
##SBATCH --mem-per-cpu=5G
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err
###SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=seh38@hi.is

#load modules

module load Stages/2023
module load GCCcore/.11.3.0  
module load GCC/11.3.0  
module load OpenMPI/4.1.4
module load TensorFlow/2.11.0-CUDA-11.7
module load scikit-learn/1.1.2
module load cuDNN/8.6.0.163-CUDA-11.7
module load CUDA/11.7
module load NCCL/default-CUDA-11.7
module load matplotlib/3.5.2
module load Python/3.10.4


#run Python program
srun --cpu-bind=none python -u LSTM.py