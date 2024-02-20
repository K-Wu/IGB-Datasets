#!/bin/bash
#SBATCH --time=167:59:00
#SBATCH -p a100
#SBATCH --exclusive
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --job-name=do_dist_training
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

# Load any necessary modules (e.g., Anaconda)
export PYTHONPATH=$PYTHONPATH:$MY_TOOLKIT_PATH/../

# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate gids_osdi24
which python

# Run your Python script
# MY_TOOLKIT_PATH is set in sbatch_and_tail.sh and is /benchmark
srun --ntasks=2 --ntasks-per-node=1 bash $MY_TOOLKIT_PATH/output_ip_list.sh


# If there is argument then pass it as --graph_name argument.
# Supported datasets involve mag240m, igbhlarge.
# num_gpus seems to specify number of GPUs per node.
# ip_config, and part_config need to be passed both as launcher argument and subcommand argument because the launcher will only pass them to server(s) and clients via environment variables.
# The arguments list is based on https://github.com/dmlc/dgl/blob/master/examples/distributed/graphsage/README.md#step-3-launch-distributed-jobs

python -m benchmark.slurm_launcher --num_trainers=1 --num_samplers=0 --num_servers=1 --ip_config=/tmp/node_lists.$SLURM_JOB_ID.out --nodename_config=/tmp/node_name_lists.$SLURM_JOB_ID.out --workspace=$MY_TOOLKIT_PATH/../workspace --part_config=$MY_TOOLKIT_PATH/../out_dataset/igb240m_medium_2_1/igb240m_medium.json "python -m benchmark.do_graphsage_node_classification --ip_config=/tmp/node_lists.$SLURM_JOB_ID.out --num_gpus=1 --part_config=$MY_TOOLKIT_PATH/../out_dataset/igb240m_medium_2_1/igb240m_medium.json --graph_name=igb240m_medium"
