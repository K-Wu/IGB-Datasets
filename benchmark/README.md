# Partitioning
The computer needs 1TB memory to partition the IGBH-full graph. The partition script by default uses random partitioning.
```
python -m benchmark.do_partition_graph --num_parts=2 --num_trainers_per_machine=2 --memory_efficient_impl
```