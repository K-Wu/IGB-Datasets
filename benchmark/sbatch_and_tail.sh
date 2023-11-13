#!/bin/bash

# Submit the job and capture the output
# Get script name from the argument
output=$(sbatch $1)

# Extract the job ID from the output
job_id=$(echo $output | awk '{print $4}')

# Print the job ID
echo "Submitted job with ID: $job_id"
# Now you can use $job_id for further processing if needed
sleep 5
tail -f my_output.$job_id.out -f my_output.$job_id.err