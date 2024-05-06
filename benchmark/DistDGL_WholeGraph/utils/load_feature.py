"""Utilities for loading feature tensor in [shared_mem | distributed] for distributed GNN tasks. """
from typing import Union
import os


import pylibwholegraph.torch as wgth
import torch



def _load_wholegraph_distribute_feature_tensor_homogeneous(feature_comm, feat_last_dim_size: int, feat_path: str, dataset: str, wm_feat_location: str = "cpu"):
    """Setup and initialize the wholegraph feature store."""
    embedding_wholememory_type = (
        "distributed"  # distribute feature tensor across all procs
    )
    embedding_wholememory_location = wm_feat_location
    if wgth.get_rank() == 0:
        print(
            f"wholegraph feature embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, communication backend={feature_comm.distributed_backend}"
        )
    feat_store = wgth.create_embedding_from_filelist(
        feature_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        os.path.join(feat_path, "{}_feat.bin".format(dataset)),
        torch.float,
        feat_last_dim_size,
        optimizer=None,
        cache_policy=None,
    )
    return feat_store

def _load_wholegraph_distribute_feature_tensor_heterogeneous(feature_comm, feat_dim: dict[str,list[int]], feat_path: dict[str,str], dataset: str, wm_feat_location: str = "cpu") -> tuple[wgth.WholeMemoryEmbedding, dict[str, tuple[int, int]]]:
    """Setup and initialize the wholegraph feature store."""
    embedding_wholememory_type = (
        "distributed"  # distribute feature tensor across all procs
    )
    embedding_wholememory_location = wm_feat_location
    if wgth.get_rank() == 0:
        print(
            f"wholegraph feature embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, communication backend={feature_comm.distributed_backend}"
        )

    feat_last_dim_size = feat_dim[list(feat_dim.keys())[0]][-1]
    num_features = sum([feat_dim[node_type][0] for node_type in feat_dim])

    num_features_offset: dict[str, tuple[int,int]] = {list(feat_dim.keys())[idx_type]: (sum([feat_dim[node_type_][0] for node_type_ in list(feat_dim.keys())[:idx_type]]),sum([feat_dim[node_type_][0] for node_type_ in list(feat_dim.keys())[:idx_type+1]])) for idx_type in range(len(feat_dim))}
    
    root_feat_store = wgth.create_embedding(
        feature_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        torch.float,
        [num_features, feat_last_dim_size],
        optimizer=None,
        cache_policy=None,
    )

    for node_type in feat_dim:
        feat_tensor = root_feat_store.get_embedding_tensor().get_sub_tensor(
            [num_features_offset[node_type][0], -1],[num_features_offset[node_type][1], -1])
        feat_tensor.from_filelist(
            os.path.join(feat_path[node_type], "{}_{}_feat.bin".format(dataset, node_type))
        )
        
    return (root_feat_store, num_features_offset)

def load_wholegraph_distribute_feature_tensor(feature_comm, feat_dim: Union[dict[str,list[int]], int], feat_path: Union[dict[str, str], str], dataset: str, wm_feat_location: str = "cpu") -> Union[wgth.WholeMemoryEmbedding, tuple[wgth.WholeMemoryEmbedding,dict[str, tuple[int,int]]]]:
    if isinstance(feat_path, str):
        assert isinstance(feat_dim, int)
        return _load_wholegraph_distribute_feature_tensor_homogeneous(feature_comm, feat_dim, feat_path, dataset, wm_feat_location)
    else:
        assert isinstance(feat_dim, dict)
        return _load_wholegraph_distribute_feature_tensor_heterogeneous(feature_comm, feat_dim, feat_path, dataset, wm_feat_location)

# Commented this function because it is not used.
# import numpy
# import torch.distributed as dist
# from .distributed_launch import get_local_root, local_share
# def load_dgl_feature_tensor(feat_dim, feat_path):
#     """Load the feature tensor into shared memory."""
#     if dist.get_rank() == get_local_root():
#         print("Loading feature data")
#         with open(os.path.join(feat_path, "ogbn-papers100M_feat.bin"), "rb") as f:
#             node_feat_tensor = torch.from_numpy(
#                 numpy.fromfile(f, dtype=numpy.float32)
#             ).reshape(-1, feat_dim)
#     else:
#         node_feat_tensor = None
#     node_feat_tensor = local_share(
#         node_feat_tensor
#     )  # materialize feat tensor (in shared memory) for non-root processes
#     return node_feat_tensor
