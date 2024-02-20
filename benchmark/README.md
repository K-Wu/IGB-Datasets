# Partitioning
The computer needs 1TB memory to partition the IGBH-full graph. The partition script by default uses random partitioning.
```
python -m benchmark.do_partition_graph --num_parts=2 --num_trainers_per_machine=2 --memory_efficient_impl
```

## Cheatsheet
### Command to Check Node Memory Size
```
sinfo -N -l
```

## Reference
### python launcher in slurm script
https://github.com/microsoft/DeepSpeed/blob/4d866bd55a6b2b924987603b599c1f8f35911c4b/deepspeed/launcher/multinode_runner.py#L344