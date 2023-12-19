#!/bin/bash
# From https://stackoverflow.com/a/53759961
#SBATCH --nodes=1
#SBATCH --time=167:59:00
#SBATCH -p interlagos
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --job-name=task_parallelism
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

python $MY_TOOLKIT_PATH/playground/wait_10.py &
python $MY_TOOLKIT_PATH/playground/wait_10.py &

wait
# Check parallelism by executing sacct -j<jobid> --format=JobID,Start,End,Elapsed,NCPUS
# The two are executed in parallel, meaning python execution won't be contrained by the number of tasks.