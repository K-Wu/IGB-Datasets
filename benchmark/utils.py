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

def get_igbh_config() ->argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/u/kunwu2/projects/IGB-datasets/igb/igbh', #All files are in igbh/full/processed
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='full',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--dummy_feats', type=int, default=0, 
        choices=[0, 1], help='0:use actual feature, 1:use dummy feature')
    parser.add_argument('--synthetic', type=int, default=1,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--all_in_edges', type=bool, default=True, 
        help="Set to false to use default relation. Set this option to True to use all the relation types in the dataset since DGL samplers require directed in edges.")
    args = parser.parse_args()

    # dataset = IGBHeteroDGLDataset(args)
    # g = dataset[0]
    # print(g)
    # homo_g = dgl.to_homogeneous(g)
    # print()
    # print(homo_g)
    # print()

    return args