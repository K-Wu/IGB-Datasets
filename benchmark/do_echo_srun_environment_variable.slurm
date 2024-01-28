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

ENV_VAR_TEST_SET="hello_world" srun -n 1 bash $MY_TOOLKIT_PATH/playground/echo_environment_variable.sh