import re
from typing import List

def split_single_line(line:str) -> List[str]:
    line_patterns = [".*? \d*: Part \d \| Epoch \d* \| Step \d* \| Loss [\d.]* \| Train Acc [\d.]* \| Speed \(samples\/sec\) [\d.]* \| GPU [\d.]* MB \| Mean step time [\d.]* s",
    ".*? \d*: \[DistDataLoader _request_nest_batch\] Sampling: [\d\.]* sec, Aggregation time: [\d\.]* sec",
    ".*? \d*: \[sample_blocks\] Sampling: [\d\.]* sec, Aggregation: [\d\.]* sec",
    ".*? \d*: Part \d \| Epoch \d* \| Step \d* \| Sample \+ Aggregation Time [\d.]* sec \| Movement Time [\d.]* sec \| Train Time [\d.]* sec"]
    aggregated_pattern = "|".join(line_patterns)
    return re.findall(aggregated_pattern, line)

def split_lines(lines: List[str]) -> List[str]:
    result = []
    for line in lines:
        result.extend(split_single_line(line))
    return result

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
        lines = fd.readlines()
    lines = split_lines(lines)
    for line in lines:
        try:
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
        except Exception as err:
            print(f"[{err}] Unable to parse: {line}")

    # Calculate the average metrics for the specified epoch and number of steps
    avg_metrics: dict[tuple[str, int], dict[str, float]] = dict()
    for (node, partition_id) in part_epoch_step_logs:
        last_step_idx = max(part_epoch_step_logs[(node, partition_id)][epoch_to_collect].keys())
        sampling_times = []
        aggregation_times = []
        sample_aggregation_times = []
        movement_times = []
        train_times = []
        for idx_step in range(last_step_idx, last_step_idx - num_steps_to_collect_at_the_end, -1):
            try:
                sampling_times.append(part_epoch_step_logs[(node, partition_id)][epoch_to_collect][idx_step]['sampling_time'])
            except Exception as e:
                print(f"[Unknown key] sampling_time:", (node, partition_id), epoch_to_collect, idx_step)
            try:
                aggregation_times.append(part_epoch_step_logs[(node, partition_id)][epoch_to_collect][idx_step]['aggregation_time'])
            except:
                print(f"[Unknown key] aggregation_time:", (node, partition_id), epoch_to_collect, idx_step)
            try:
                sample_aggregation_times.append(part_epoch_step_logs[(node, partition_id)][epoch_to_collect][idx_step]['sample_aggregation_time'])
            except:
                print(f"[Unknown key] sample_aggregation_time:", (node, partition_id), epoch_to_collect, idx_step)
            try:
                movement_times.append(part_epoch_step_logs[(node, partition_id)][epoch_to_collect][idx_step]['movement_time'])
            except:
                print(f"[Unknown key] movement_time:", (node, partition_id), epoch_to_collect, idx_step)
            try:
                train_times.append(part_epoch_step_logs[(node, partition_id)][epoch_to_collect][idx_step]['train_time'])
            except:
                print(f"[Unknown key] train_time:", (node, partition_id), epoch_to_collect, idx_step)
        avg_metrics[(node, partition_id)] = dict()
        avg_metrics[(node, partition_id)]['sampling_time'] = sum(sampling_times) / len(sampling_times)
        avg_metrics[(node, partition_id)]['aggregation_time'] = sum(aggregation_times) / len(aggregation_times)
        avg_metrics[(node, partition_id)]['sample_aggregation_time'] = sum(sample_aggregation_times) / len(sample_aggregation_times)
        avg_metrics[(node, partition_id)]['movement_time'] = sum(movement_times) / len(movement_times)
        avg_metrics[(node, partition_id)]['train_time'] =  sum(train_times) / len(train_times)
        print(f"Node: {node}, Partition: {partition_id}, Epoch: {epoch_to_collect}, Avg. Sampling Time: {avg_metrics[(node, partition_id)]['sampling_time']} sec, Avg. Aggregation Time: {avg_metrics[(node, partition_id)]['aggregation_time']} sec, Avg. Sample + Aggregation Time: {avg_metrics[(node, partition_id)]['sample_aggregation_time'] } sec, Movement Time: {avg_metrics[(node, partition_id)]['movement_time'] } sec, Avg. Train Time: {avg_metrics[(node, partition_id)]['train_time']} sec")
    
    # Print the average metrics across all partitions
    avg_sampling_times = []
    avg_aggregation_times = []
    avg_sample_aggregation_times = []
    avg_movement_times = []
    avg_train_times = []
    for (node, partition_id) in avg_metrics:
        avg_sampling_times.append(avg_metrics[(node, partition_id)]['sampling_time'])
        avg_aggregation_times.append(avg_metrics[(node, partition_id)]['aggregation_time'])
        avg_sample_aggregation_times.append(avg_metrics[(node, partition_id)]['sample_aggregation_time'])
        avg_movement_times.append(avg_metrics[(node, partition_id)]['movement_time'])
        avg_train_times.append(avg_metrics[(node, partition_id)]['train_time'])
    print(f"Overall Avg. Sampling Time: {sum(avg_sampling_times) / len(avg_sampling_times)} sec, Overall Avg. Aggregation Time: {sum(avg_aggregation_times) / len(avg_aggregation_times)} sec, Overall Avg. Sample + Aggregation Time: {sum(avg_sample_aggregation_times) / len(avg_sample_aggregation_times)} sec, Overall Movement Time: {sum(avg_movement_times) / len(avg_movement_times)} sec, Overall Avg. Train Time: {sum(avg_train_times) / len(avg_train_times)} sec")



if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    extract_timing(filename)