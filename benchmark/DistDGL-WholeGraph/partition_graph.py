import os
import json
import argparse
import time

import dgl
import torch as th
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset


def load_reddit(self_loop=True):
    """Load reddit dataset."""
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    """Load ogbn dataset."""
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph, num_labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--root-dir",
        default="./",
        help="graph and feature dataset root directory.",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--use-wm",
        action="store_true",
        help="turn the features into wholegraph compatible format.",
    )
    argparser.add_argument(
        "--keep-dgl-features",
        action="store_true",
        help="whether backup the features in distDGL format if use-wm is True.",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit()
    elif args.dataset in ["ogbn-products", "ogbn-papers100M"]:
        g, _ = load_ogb(args.dataset, args.root_dir)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    if args.use_wm:
        wg_folder = os.path.join(args.output, 'wg_features')
        if not os.path.exists(wg_folder):
            os.makedirs(wg_folder)  
                  
        if args.keep_dgl_features:
            wg_feat = g.ndata["features"].to(dtype=th.float)
        else:
            wg_feat = g.ndata.pop("features").to(dtype=th.float)

    orig_nids, orig_eids = dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        return_mapping=True,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )

    # dump features in binary format for wholegraph
    if args.use_wm:
        assert len(orig_nids) == len(wg_feat)
        print("Reordering raw features for WholeGraph...")
        wg_feat = wg_feat[orig_nids]

        # append "wholegraph" for part_config
        part_config = os.path.join(args.output, "{}.json".format(args.dataset))
        assert os.path.exists(part_config)
        with open(part_config, "r") as file:
            part_meta = json.load(file)
        part_meta["feature_store"] = {"WholeGraph": os.path.relpath(wg_folder, args.output)}
        with open(part_config, "w") as outfile:
            json.dump(part_meta, outfile, sort_keys=False, indent=4)

        # load wholegraph feat
        wg_metadata = {}
        wg_metadata["features"] = {"shape": list(wg_feat.shape), "dtype": str(wg_feat.dtype)}

        with open(
            os.path.join(wg_folder, "{}_feat.json".format(args.dataset)), "w"
        ) as f:
            json.dump(wg_metadata, f)                      

        with open(
            os.path.join(wg_folder, "{}_feat.bin".format(args.dataset)), "wb"
        ) as f:
            print("Saving node feature to binary file...")
            wg_feat.numpy().tofile(f)