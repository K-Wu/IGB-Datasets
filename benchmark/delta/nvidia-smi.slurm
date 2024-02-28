#!/bin/bash
#SBATCH --time=10:59:00
#SBATCH -p gpuA100x4
#SBATCH --account=bcdt-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=do_dist_training
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err


nvidia-smi