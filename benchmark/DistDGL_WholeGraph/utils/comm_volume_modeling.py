# The features are equally divided and loaded to each GPUs, according to https://github.com/rapidsai/wholegraph/blob/4ff587121688336c589925425d9e58f2b5393da8/cpp/src/wholememory/file_io.cpp#L300

import pylibwholegraph.torch as wgth


def get_communication_volume(is_heterogeneous: bool, input_nodes, num_nodes) -> tuple[int, int]:
    """Calculate the communication volume of the input nodes. Returns the number of local and remote communication volume (in the number of node features)."""
    if is_heterogeneous:
        raise NotImplementedError("Heterogeneous graph is not supported yet.")
    
    rank = wgth.get_rank()
    world_size = wgth.get_world_size()
    local_node_feature_beg = rank * num_nodes // world_size
    local_node_feature_end = (rank + 1) * num_nodes // world_size

    # For each node in input_nodes, if it is in the local chunk, it adds local communication volume; otherwise, it adds remote communication volume.
    local_communication_volume = 0
    remote_communication_volume = 0
    for node in input_nodes:
        if node >= local_node_feature_beg and node < local_node_feature_end:
            local_communication_volume += 1
        else:
            remote_communication_volume += 1
    
    return local_communication_volume, remote_communication_volume