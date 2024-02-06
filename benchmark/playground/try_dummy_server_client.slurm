#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --time=167:59:00
#SBATCH -p rome
#SBATCH --exclusive
# #SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=partition_graph
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err
# #SBATCH --sockets-per-node=2

hostname=$(hostname)
# Replace $hostname with $hostname-ib0 to get ib0 address in the following line
thisip=$(getent ahostsv4 $hostname | grep STREAM | head -n 1  | awk '{ print $1 }')

echo "thisip: $thisip"

echo $MY_TOOLKIT_PATH
srun -ln1 --mem-per-cpu 1000MB python $MY_TOOLKIT_PATH/playground/dummy_server.py 127.0.0.1 &
sleep 5
srun -ln1 --mem-per-cpu 1000MB python $MY_TOOLKIT_PATH/playground/dummy_client.py 127.0.0.1
