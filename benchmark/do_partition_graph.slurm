#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=167:59:00
#SBATCH -p rome
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --job-name=partition_graph
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

# Load any necessary modules (e.g., Anaconda)


# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate gids_osdi24
which conda
which python

# Run your Python script
python -m benchmark.do_partition_graph