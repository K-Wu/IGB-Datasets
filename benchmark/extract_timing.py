

def extract_timing(filename: str,epoch_to_collect = 1, num_steps_to_collect_at_the_end = 100):
    """
    Extract timing from the following two formatted strings in sampled steps in the output:
    hydro04 3: [DistDataLoader _request_nest_batch] Sampling: 0.006553173065185547 sec, Aggregation time: 0.04140210151672363 sec
    hydro04 3: Part 3 | Epoch 00001 | Step 00001 | Sample + Aggregation Time 0.1364 sec | Movement Time 0.4427 sec | Train Time 0.1685 sec

    And store them in a dictionary with the following structure:
    part_epoch_step_logs: dict[tuple[str, int], dict[int, dict[int, dict]]] = {
    ('hydro04', 3): {
        1: {
            1: {
                'sampling_time': 0.006553173065185547,
                'aggregation_time': 0.04140210151672363,
                'sample_aggregation_time': 0.1364,
                'movement_time': 0.4427,
                'train_time': 0.1685
            }
        }
    }
    """
    part_epoch_step_logs: dict[tuple[str, int], dict[int, dict[int, dict]]] = dict()
    part_unknown_epoch_step_logs: dict[tuple[str, int], dict] = dict() # The epoch and step is unknown when the line is [DistDataLoader _request_nest_batch]. We need to wait for the next line to get the epoch.
    with open(filename) as fd:
        for line in fd:
            if line.find("_request_nest_batch]") != -1:
                # Pattern: "<nodename> <partition_id>: [DistDataLoader _request_nest_batch] Sampling: <sampling_time> sec, Aggregation time: <aggregation_time> sec"
                parts = line.strip().split()
                node = parts[0]
                partition_id = int(parts[1][:-1])
                sampling_time = float(parts[-6])
                aggregation_time = float(parts[-2])
                if (node, partition_id) not in part_unknown_epoch_step_logs:
                    part_unknown_epoch_step_logs[(node, partition_id)] = dict()
                part_unknown_epoch_step_logs[(node, partition_id)]['sampling_time'] = sampling_time
                part_unknown_epoch_step_logs[(node, partition_id)]['aggregation_time'] = aggregation_time
            elif line.find("Movement Time") != -1:
                # Pattern: "<nodename> <partition_id>: Part <partition_id> | Epoch <epoch_idx> | Step <step_idx> | Sample + Aggregation Time <sampling_and_aggregation_time> sec | Movement Time <movement_time> sec | Train Time <training_time> sec"
                parts = line.strip().split()
                node = parts[0]
                partition_id = int(parts[1][:-1])
                epoch_idx = int(parts[6])
                step_idx = int(parts[9])
                sample_aggregation_time = float(parts[15])
                movement_time = float(parts[20])
                train_time = float(parts[25])
                if (node, partition_id) not in part_epoch_step_logs:
                    part_epoch_step_logs[(node, partition_id)] = dict()
                if epoch_idx not in part_epoch_step_logs[(node, partition_id)]:
                    part_epoch_step_logs[(node, partition_id)][epoch_idx] = dict()
                if step_idx not in part_epoch_step_logs[(node, partition_id)][epoch_idx]:
                    part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx] = dict()
                part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['sampling_time'] = part_unknown_epoch_step_logs[(node, partition_id)]['sampling_time']
                part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['aggregation_time'] = part_unknown_epoch_step_logs[(node, partition_id)]['aggregation_time']
                part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['sample_aggregation_time'] = sample_aggregation_time
                part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['movement_time'] = movement_time
                part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['train_time'] = train_time
                part_unknown_epoch_step_logs.pop((node, partition_id))

    # Calculate the average metrics for the specified epoch and number of steps
    for (node, partition_id) in part_epoch_step_logs:
        last_step_idx = max(part_epoch_step_logs[(node, partition_id)][epoch_to_collect].keys())
        sampling_time = 0
        aggregation_time = 0
        sample_aggregation_time = 0
        movement_time = 0
        train_time = 0
        num_steps = 0
        for i in range(last_step_idx, last_step_idx - num_steps_to_collect_at_the_end, -1):
            sampling_time += part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx - i]['sampling_time']
            aggregation_time += part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx - i]['aggregation_time']
            sample_aggregation_time += part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx - i]['sample_aggregation_time']
            movement_time += part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx - i]['movement_time']
            train_time += part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx - i]['train_time']
            num_steps += 1
        part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['sampling_time'] = sampling_time / num_steps
        part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['aggregation_time'] = aggregation_time / num_steps
        part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['sample_aggregation_time'] = sample_aggregation_time / num_steps
        part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['movement_time'] = movement_time / num_steps
        part_epoch_step_logs[(node, partition_id)][epoch_idx][step_idx]['train_time'] = train_time / num_steps
        print(f"Node: {node}, Partition: {partition_id}, Epoch: {epoch_idx}, # Steps: {num_steps}, Avg. Sampling Time: {sampling_time / num_steps} sec, Avg. Aggregation Time: {aggregation_time / num_steps} sec, Avg. Sample + Aggregation Time: {sample_aggregation_time / num_steps} sec, Movement Time: {movement_time / num_steps} sec, Avg. Train Time: {train_time / num_steps} sec")



if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    extract_timing(filename)