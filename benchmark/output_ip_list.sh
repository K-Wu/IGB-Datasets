function output_all_ips {
# Get node lists from SLURM_JOB_NODELIST
# e.g., SLURM_JOB_NODELIST is hydro[05-06]
# We obtain node_lists as hydro05 hydro06
node_lists=$(scontrol show hostnames $SLURM_JOB_NODELIST)
node_lists=(${node_lists[@]})
echo "node lists ${node_lists[@]}"
echo "the first in node lists ${node_lists[0]}"


# Get the hostname of the node
hostname=$(hostname)

# Get the IP address of the node.
# This gets the first STREAM ip address
# From https://www.linuxtutorials.org/resolve-hostname-to-ip-address-linux/
ip=$(getent ahostsv4 $hostname | grep STREAM | head -n 1  | awk '{ print $1 }')
ips=()
# Get the IP address of each node in node_lists
for node in ${node_lists[@]}
do
    ips+=($(getent ahostsv4 $node | grep STREAM | head -n 1  | awk '{ print $1 }'))
done

# Store ips into  /tmp/node_lists.%j.out
for ip in ${ips[@]}
do
    echo $ip >> /tmp/node_lists.$SLURM_JOB_ID.out
done

echo "hostname $(hostname) The IP address of the node is $ip."
echo "ips ${ips[@]}"

echo "Content in /tmp/node_lists.$SLURM_JOB_ID.out"
cat /tmp/node_lists.$SLURM_JOB_ID.out
}
output_all_ips