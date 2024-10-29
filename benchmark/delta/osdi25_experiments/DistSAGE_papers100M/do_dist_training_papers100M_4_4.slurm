#!/bin/bash
#SBATCH --time=1:59:00
#SBATCH -p gpuA100x4
#SBATCH --account=bcdt-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --job-name=do_dist_training
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

# Load any necessary modules (e.g., Anaconda)
export PYTHONPATH=$PYTHONPATH:$MY_TOOLKIT_PATH/../

# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate ssd_gnn_25
which python

# Run your Python script
# MY_TOOLKIT_PATH is set in sbatch_and_tail.sh and is /benchmark
srun --ntasks=4 --ntasks-per-node=1 bash $MY_TOOLKIT_PATH/output_ip_list_delta.sh


# If there is argument then pass it as --graph_name argument.
# Supported datasets involve mag240m, igbhlarge.
# num_gpus seems to specify number of GPUs per node.
# ip_config, and part_config need to be passed both as launcher argument and subcommand argument because the launcher will only pass them to server(s) and clients via environment variables.
# The arguments list is based on https://github.com/dmlc/dgl/blob/master/examples/distributed/graphsage/README.md#step-3-launch-distributed-jobs

export NVSHMEM_NVTX=common
# nsys profile -o igb_medium_4_4_with_wg --force-overwrite true --nic-metrics=true --trace=nvtx,cuda --duration=600 
python -m benchmark.DistDGL_WholeGraph.launch --num_trainers=4 --num_samplers=0 --num_servers=1 --ip_config=/tmp/node_lists.$SLURM_JOB_ID.out --nodename_config=/tmp/node_name_lists.$SLURM_JOB_ID.out --workspace=$MY_TOOLKIT_PATH/../workspace --part_config=$MY_TOOLKIT_PATH/../out_ogbn-papers100M_4_4/ogbn-papers100M.json "python -m benchmark.do_graphsage_node_classification --ip_config=/tmp/node_lists.$SLURM_JOB_ID.out --num_gpus=4 --part_config=$MY_TOOLKIT_PATH/../out_ogbn-papers100M_4_4/ogbn-papers100M.json --graph_name=ogbn-papers100M --model=DistSAGE --num_hidden=128 --n_layers=3 --fan_out=10,5,5 --batch_size=256 --use-gb --gb-cache-size=1048576 --lr=0.01"
