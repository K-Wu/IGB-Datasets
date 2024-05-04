#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=167:59:00
#SBATCH -p interlagos
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
which python

# Run your Python script

# If there is argument then pass it as --dataset
# supported datasets involve mag240m, igbhlarge
catchsegv python -m benchmark.heterogeneous_version.do_partition_graph --num_parts=2 --num_trainers_per_machine=2 --dataset=igbhmedium --output=out_single_dataset_heterogeneous_version_igbhmedium_2_2