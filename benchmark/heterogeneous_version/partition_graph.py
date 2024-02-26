# Adapted from https://github.com/dmlc/dgl/blob/7c51cd16436c2d774be63c0cec8f222dadf01148/examples/pytorch/rgcn/experimental/partition_graph.py
# Compared with the do_partition_graph.py and do_graphsage_node_classification.py, this file 1) work for ogb academic heterogeneous graphs, i.e., paper100M and mag, 2) assign masks to paper, and 3) use original DGL partition API instead of our own memory efficient one. 

import argparse
import time

import dgl
import numpy as np
import torch as th

from ogb.nodeproppred import DglNodePropPredDataset
from ..do_partition_graph import _load_igbh

def load_igbh_medium():
    g = _load_igbh("medium")
    # Masks are already in the graph. We only need to canonicalize the feature and label names.
    for ntype in ["author", "paper", "institute", "fos"]:
        # Rename the features and labels to the canonical names.
        if "feat" in g.nodes[ntype].data:
            g.nodes[ntype].data["features"] = g.nodes[ntype].data.pop('feat')
        if "label" in g.nodes[ntype].data:
            g.nodes[ntype].data["labels"] = g.nodes[ntype].data.pop('label')
    return g


def load_ogb_lsc_mag_240m():
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset()
    '''
    edge_index is numpy.ndarray of shape (2, num_edges).
    - first row: indices of source nodes (indexed by source node types)
    - second row: indices of target nodes (indexed by target node types)
    In other words, i-th edge connects from edge_index[0,i] to edge_index[1,i].
    '''
    edge_index_writes = dataset.edge_index('author', 'writes', 'paper') 
    edge_index_cites = dataset.edge_index('paper', 'paper')
    edge_index_affiliated_with = dataset.edge_index('author', 'institution')
    print("Constructing graph_data", flush=True)
    graph_data = {
        ('author', 'affiliated_with', 'institute'): (edge_index_affiliated_with[0,:], edge_index_affiliated_with[1,:]),
        ('paper', 'cites', 'paper'): (edge_index_cites[0,:], edge_index_cites[1,:]),
        ('author', 'writes', 'paper'): (edge_index_writes[ 0, :], edge_index_writes[ 1,:]),
    }
    print("Constructed graph_data", flush=True)

    num_nodes_dict = {'paper': dataset.num_papers, 'author': dataset.num_authors, 'institute': dataset.num_institutions}

    graph = dgl.heterograph(graph_data, num_nodes_dict)  
    print("Created heterograph", flush=True)

    split_dict = dataset.get_idx_split()
    train_idx = split_dict['train']
    val_idx = split_dict['valid']
    test_idx = split_dict['test-whole']
    train_mask = th.zeros((graph.num_nodes("paper"),), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((graph.num_nodes("paper"),), dtype=th.bool)
    val_mask[val_idx] = True
    test_mask = th.zeros((graph.num_nodes("paper"),), dtype=th.bool)
    test_mask[test_idx] = True
    graph.nodes["paper"].data["train_mask"] = train_mask
    graph.nodes["paper"].data["val_mask"] = val_mask
    graph.nodes["paper"].data["test_mask"] = test_mask

    graph.nodes["paper"].data["labels"] = th.tensor(dataset.paper_label)
    graph.nodes["paper"].data["features"] = th.tensor(dataset.paper_feat, dtype=th.float32)

    # TODO: other features

    return graph, dataset.num_classes


def load_ogb(dataset):
    if dataset == "ogbn-mag":
        dataset = DglNodePropPredDataset(name=dataset)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]["paper"]
        val_idx = split_idx["valid"]["paper"]
        test_idx = split_idx["test"]["paper"]
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], "rev-" + etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes["paper"].data["feat"] = hg_orig.nodes["paper"].data["feat"]
        paper_labels = labels["paper"].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        category = "paper"
        print("Number of relations: {}".format(num_rels))
        print("Number of class: {}".format(num_classes))
        print("Number of train: {}".format(len(train_idx)))
        print("Number of valid: {}".format(len(val_idx)))
        print("Number of test: {}".format(len(test_idx)))

        # get target category id
        category_id = len(hg.ntypes)
        for i, ntype in enumerate(hg.ntypes):
            if ntype == category:
                category_id = i

        train_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        train_mask[train_idx] = True
        val_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        val_mask[val_idx] = True
        test_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        test_mask[test_idx] = True
        hg.nodes["paper"].data["train_mask"] = train_mask
        hg.nodes["paper"].data["val_mask"] = val_mask
        hg.nodes["paper"].data["test_mask"] = test_mask

        hg.nodes["paper"].data["labels"] = paper_labels
        return hg
    else:
        raise ("Do not support other ogbn datasets.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset", type=str, default="mag240m", help="datasets: ogbn-mag, mag240m, igbhmedium"
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="random", help="the partition method: metis or random"
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
        default="out_single_dataset_heterogeneous_version",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "mag240m":
        g, num_classes = load_ogb_lsc_mag_240m()
    elif args.dataset == "igbhmedium":
        g = load_igbh_medium()
    else:
        g = load_ogb(args.dataset)

    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start), flush=True
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()), flush=True)
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.nodes["paper"].data["train_mask"]),
            th.sum(g.nodes["paper"].data["val_mask"]),
            th.sum(g.nodes["paper"].data["test_mask"]),
        ), flush=True
    )

    if args.balance_train:
        balance_ntypes = {"paper": g.nodes["paper"].data["train_mask"]}
    else:
        balance_ntypes = None
    start = time.time()
    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )
    print("Partitioning takes {:.3f} seconds".format(time.time() - start), flush=True)