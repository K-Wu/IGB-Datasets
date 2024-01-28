import sys
from .. import utils
import os
sys.path.insert(1, os.path.join(utils.get_git_root_path()))
import igb.utils
from dgl.convert import graph
import torch
import numpy as np


def test_cat_mmap():
    srcs = [torch.tensor([1,2,3]), torch.tensor([4,5,6])]
    dsts = [torch.tensor([7,8,9]), torch.tensor([10,11,12])]
    retg = graph(
    (igb.utils.cat_mmap_arrays(srcs), igb.utils.cat_mmap_arrays(dsts)),
    )
    return retg



def test_create_graph_from_mmap_load():
    srcs = np.array([1,2,3])
    dsts = np.array([7,8,9])
    src_filename = igb.utils.get_unique_temp_file_name()
    dst_filename = igb.utils.get_unique_temp_file_name()
    np.save(src_filename, srcs)
    np.save(dst_filename, dsts)
    srcs_reloaded = np.load(src_filename+".npy", mmap_mode='r')
    dsts_reloaded = np.load(dst_filename+".npy", mmap_mode='r')
    retg = graph((srcs_reloaded, dsts_reloaded))
    return retg


if __name__ == "__main__":
    assert utils.is_pwd_correct_for_benchmark(), (
        "Please run this script at the repository root path."
        " The command will be something like python -m"
        " benchmark.playground.test_mmap_load"
    )
    print(test_cat_mmap())
    print(test_create_graph_from_mmap_load())