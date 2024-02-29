"""Utilities for loading feature tensor in [shared_mem | distributed] for distributed GNN tasks. """

import os

import numpy

import pylibwholegraph.torch as wgth
import torch
import torch.distributed as dist

from .distributed_launch import get_local_root, local_share


def load_wholegraph_distribute_feature_tensor(feature_comm, feat_dim, feat_path, args):
    """Setup and initialize the wholegraph feature store."""
    embedding_wholememory_type = (
        "distributed"  # distribute feature tensor across all procs
    )
    embedding_wholememory_location = args.wm_feat_location
    if wgth.get_rank() == 0:
        print(
            f"wholegraph feature embedding: type={embedding_wholememory_type}, location={embedding_wholememory_location}, communication backend={feature_comm.preferred_distributed_backend}"
        )
    feat_store = wgth.create_embedding_from_filelist(
        feature_comm,
        embedding_wholememory_type,
        embedding_wholememory_location,
        os.path.join(feat_path, "ogbn-papers100M_feat.bin"),
        torch.float,
        feat_dim,
        optimizer=None,
        cache_policy=None,
    )
    return feat_store


def load_dgl_feature_tensor(feat_dim, feat_path):
    """Load the feature tensor into shared memory."""
    if dist.get_rank() == get_local_root():
        print("Loading feature data")
        with open(os.path.join(feat_path, "ogbn-papers100M_feat.bin"), "rb") as f:
            node_feat_tensor = torch.from_numpy(
                numpy.fromfile(f, dtype=numpy.float32)
            ).reshape(-1, feat_dim)
    else:
        node_feat_tensor = None
    node_feat_tensor = local_share(
        node_feat_tensor
    )  # materialize feat tensor (in shared memory) for non-root processes
    return node_feat_tensor
