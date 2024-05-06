"""Utilities for launching wholegraph distributed tasks. """
import os
from typing import Union
import os
import json
import pylibwholegraph.torch as wgth

import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch.wholememory_ops as wm_ops

class wholegraph_config:
    """Add/initialize default options required for distributed launch incorprating with wholegraph

    NOTE: This class might be deprecated soon once wholegraph's update its configuration API.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.launch_env_name_world_rank = "RANK"
        self.launch_env_name_world_size = "WORLD_SIZE"
        self.launch_env_name_master_addr = "MASTER_ADDR"
        self.launch_env_name_master_port = "MASTER_PORT"
        self.launch_env_name_local_size = "LOCAL_WORLD_SIZE"
        self.launch_env_name_local_rank = "LOCAL_RANK"
        if self.launch_agent == "mpi":
            self.master_addr = "" # pick from env var
            self.master_port = -1 # pick from env var
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if "LOCAL_WORLD_SIZE" in os.environ:
            self.local_size = int(os.environ["LOCAL_WORLD_SIZE"])

        # make sure the following arguments are avail for wholegraph
        assert self.local_rank is not None
        assert self.local_size is not None and self.local_size > 0

# initialize wholegraph and return its global communicator
def init_wholegraph(args):
    config = wholegraph_config(launch_agent=args.wg_launch_agent, local_size=args.ngpu_per_node)
    wgth.distributed_launch(config, lambda: None)
    wmb.init(0)
    wgth.comm.set_world_info(wgth.get_rank(), wgth.get_world_size(), wgth.get_local_rank(), wgth.get_local_size(),)
    if args.wg_launch_agent == 'pytorch':
        assert args.wg_comm_backend=='nccl', "nvshmem does not support launching through pytorch. Please use mpi instead."
    global_comm = wgth.comm.get_global_communicator(args.wg_comm_backend)
    return global_comm

def get_heterogeneous_node_num(config_path) -> dict[str, int]:
    with open(config_path) as f:
        metadata = json.load(f)
        assert "num_nodes" in metadata, f"Key 'num_nodes' needs to be in the config file for using wholegraph feature store."
        num_nodes = {}
        for node_type in metadata["node_map"]:
            num_nodes[node_type] = 0
            for range_ in metadata["node_map"][node_type]:
                # Each range is a list of [start, end] where start is inclusive and end is exclusive
                num_nodes[node_type] += range_[1] - range_[0]
        assert sum(num_nodes.values()) == metadata["num_nodes"], "Sum of node numbers in node_map does not match num_nodes in metadata"
        return num_nodes

# parse the config file to extract information like feature dimension and wholegraph feature file path
def parse_wholegraph_config(config_path, dataset) -> tuple[Union[dict, int], str]:
    try:
        with open(config_path) as f:
            part_metadata = json.load(f)
            assert "feature_store" in part_metadata, f"Key 'feature_store' needs to be in the config file for using wholegraph feature store."
            assert "WholeGraph" in part_metadata["feature_store"], f"Unknow feature store manager. Only support WholeGraph for now"
            config_dir = os.path.dirname(config_path)
            wg_dir = os.path.join(config_dir, part_metadata["feature_store"]["WholeGraph"])
    except Exception as e:
        # Get default location if config file does not specify
        print("Config file did not specify the wg_features directory. Using default location for wholegraph feature store.")
        wg_dir = os.path.join(config_dir, 'wg_features')
    
    assert os.path.exists(wg_dir), f"Wholegraph feature store not found at {wg_dir}"

    num_nodes = get_heterogeneous_node_num(config_path)

    if len(num_nodes) > 1:
        # Heterogeneous graph
        feat_dim: dict[str, int] = {}
        for node_type in num_nodes:
            with open(os.path.join(wg_dir, "{}_{}_feat.json".format(dataset, node_type))) as f:
                feat = json.load(f)
                for feat_name in feat:
                    assert feat_name == "features"
                    feat_dim[node_type] = feat[feat_name]["shape"][1]
        return feat_dim, wg_dir
    else:
        # Homogeneous graph
        feat_last_dim_size: int = -1
        with open(os.path.join(wg_dir, "{}_feat.json".format(dataset))) as f:
            feat = json.load(f)
            for feat_name in feat:
                assert feat_name == "features"
                # Get the last dimension as required by wholegraph
                feat_last_dim_size = feat[feat_name]["shape"][1]
        assert feat_last_dim_size > 0, "Feature dimension must be greater than 0"
        return feat_last_dim_size, wg_dir

# TODO: extend this to heterogeneous graph. This is not done yet since it is only used in evaluation and only used in the wholegraph example.
# create a new wholegraph distributed tensor
def create_wholegraph_dist_tensor(shape, dtype, location='cpu'):
    comm = wgth.comm.get_global_communicator()
    return wgth.create_embedding(
        comm,
        "distributed",
        location,
        dtype,
        shape,
        optimizer=None,
        cache_policy=None,
    )

# check if the input tensor is wholegraph embedding tensor
def is_wm_tensor(tensor):
    return isinstance(tensor, wgth.WholeMemoryEmbedding)

# scatter the values to a wholegraph embedding tensor according to a given set of indexes
def wm_scatter(update_emb, indexes, tensor):
    comm = wgth.comm.get_global_communicator()
    indexes = indexes.cuda()
    update_emb = update_emb.cuda()
    wmb_embedding_tensor = tensor.wmb_embedding.get_embedding_tensor()

    wm_ops.wholememory_scatter_functor(update_emb, indexes, wmb_embedding_tensor)
    comm.barrier()
    return
