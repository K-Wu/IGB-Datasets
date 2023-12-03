#!/bin/bash
# Usage bash benchmark/sbatch_and_tall.sh /path/to/job.slurm
# Submit the job and capture the output
# Get script name from the argument

if [ "$#" -eq 0 ]
then
    echo "Usage: bash sbatch_and_tail.sh /pato/to/job.slurm" >&2
    echo "argument passing is supported, e.g., bash sbatch_and_tail.sh do_partition_graph.slurm igbhfull" >&2
    exit 1
fi

echo "Executing command sbatch $@"

output=$(sbatch "$@")

# Extract the job ID from the output
job_id=$(echo $output | awk '{print $4}')

# Print the job ID
echo "Submitted job with ID: $job_id"

# Try to tail until the file is found
while ! (ls my_output.$job_id.out my_output.$job_id.err 2>/dev/null)
do
    echo "Waiting for output file..."
    sleep 5
done

# Now you can use $job_id for further processing if needed
tail -f my_output.$job_id.out -f my_output.$job_id.err