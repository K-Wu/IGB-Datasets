from dgl.data.utils import load_tensors, store_tensors
import argparse
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str, required=True)
    args = argparser.parse_args()

    # First make sure the path is a directory consisting of DistDGL partitions
    assert os.path.exists(args.path)
    json_files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.json')]
    metadata_exists = False
    for json_file in json_files:
        json_dict = json.load(open(json_file, 'r'))
        if "graph_name" in json_dict:
            metadata_exists = True
            break
    assert metadata_exists, "DistDGL partitions Metadata not found in the directory"

    # From https://www.tutorialspoint.com/How-to-get-a-list-of-all-sub-directories-in-the-current-directory-using-Python
    subdirs = [os.path.join(args.path, o) for o in os.listdir(args.path) if os.path.isdir(os.path.join(args.path,o))]
    # generate dummy node features for each partition
    for subdir in subdirs:
        node_feat = load_tensors(os.path.join(subdir, 'node_feats.dgl'))
        # Replace every tensor in node_feat with a tensor of the same first dimension but with 1 as the second dimension
        for key in node_feat:
            assert len(node_feat[key].shape) == 2, "Node feature tensor should have 2 dimensions"
            node_feat[key] = torch.rand((node_feat[key].shape[0], 1))
        store_tensors(os.path.join(subdir, 'node_feats_dummy.dgl'), node_feat)
