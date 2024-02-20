import sys
import os
import argparse
import subprocess
from typing import Any
from dgl import DGLHeteroGraph

def construct_graph_attributes(g_hetero: DGLHeteroGraph) -> dict[str, Any]:
    g_attrs: dict[str, Any] = {}
    g_attrs["is_homogeneous"] = g_hetero.is_homogeneous # (property)
    g_attrs["ntypes"] = g_hetero.ntypes # (iterable)
    g_attrs["canonical_etypes"] = g_hetero.canonical_etypes # (iterable)
    g_attrs["etypes"] = g_hetero.etypes # (iterable)

    g_attrs["get_ntype_id"] = {} # (dict. Keys are ntype from g.ntypes)
    g_attrs["get_etype_id"] = {} # (dict. Keys are etype from g.canonial_etypes)
    g_attrs["num_nodes"] = {} # (dict. Keys are ntype from g.ntypes, and total (""))
    g_attrs["num_edges"] = {} # (dict. Keys are etype from g.canonial_etypes, and total (""))
    g_attrs["nodes_data"] = {} # (dict. Keys are ntype from g.ntypes, and total (""))
    g_attrs["edges_data"] = {} # (dict. Keys are etype from g.canonial_etypes, and total (""))

    for ntype in g_hetero.ntypes:
        g_attrs["get_ntype_id"][ntype] = g_hetero.get_ntype_id(ntype)
        g_attrs["num_nodes"][ntype] = g_hetero.num_nodes(ntype)
        g_attrs["nodes_data"][ntype] = g_hetero.nodes[ntype].data
    for etype in g_hetero.canonical_etypes:
        g_attrs["get_etype_id"][etype] = g_hetero.get_etype_id(etype)
        g_attrs["num_edges"][etype] = g_hetero.num_edges(etype)
        g_attrs["edges_data"][etype] = g_hetero.edges[etype].data
    g_attrs["num_nodes"][""] = g_hetero.num_nodes()
    g_attrs["num_edges"][""] = g_hetero.num_edges()

    return g_attrs

def assert_git_exists() -> None:
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except Exception:  # any error means git is not installed
        raise OSError(
            "Git is not installed. Please install git and try again."
        )

def get_git_root_path() -> str:
    """Get the root path of the git repository, i.e., cupy-playground."""
    assert_git_exists()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def is_pwd_correct_for_benchmark():
    script_path: str = sys.argv[0]
    pwd: str = os.getcwd()
    repo_in_dir: str = os.path.dirname(os.path.dirname(script_path))
    return pwd == repo_in_dir

def get_numpy_file_shape(filename: str):
    import numpy as np
    return np.load(filename, mmap_mode='r').shape

# From: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def get_igbh_config() ->argparse.ArgumentParser:

    args = AttributeDict()
    args.path = '/u/kunwu2/projects/IGB-datasets/igb/igbh'
    args.dataset_size = 'full'
    args.num_classes = 19
    args.load_homo_graph=1
    args.in_memory=1
    args.dummy_feats=1
    args.synthetic=1
    args.all_in_edges=True

    # dataset = IGBHeteroDGLDataset(args)
    # g = dataset[0]
    # print(g)
    # homo_g = dgl.to_homogeneous(g)
    # print()
    # print(homo_g)
    # print()

    return args

def get_igb_config() -> argparse.ArgumentParser:
    """IGB uses the same arguments as IGBH, and use a subset of the files on disk in IGBH"""
    return get_igbh_config()