#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --time=167:59:00
#SBATCH -p rome
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --job-name=partition_graph
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

# define a function
echo $MY_TOOLKIT_PATH
srun bash $MY_TOOLKIT_PATH/output_ip_list.sh

