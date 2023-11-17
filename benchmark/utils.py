import sys
import os
import argparse

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
    if args.dummy_feats:
        print("using dummy feats")

    # dataset = IGBHeteroDGLDataset(args)
    # g = dataset[0]
    # print(g)
    # homo_g = dgl.to_homogeneous(g)
    # print()
    # print(homo_g)
    # print()

    return args