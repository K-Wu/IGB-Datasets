# From https://github.com/dmlc/dgl/blob/master/examples/distributed/graphsage/partition_graph.py
import argparse
import time

import dgl
import torch as th

from dgl.convert import to_homogeneous
from .utils import get_igbh_config, is_pwd_correct_for_benchmark
from igb.dataloader import IGBHeteroDGLDatasetMassive

def _load_igbh(dataset: str):
    args = get_igbh_config()
    if dataset=="large":
        args.dataset_size="large"
    elif dataset=="full":
        pass
    else:
        raise ValueError(f"Unknown igbh dataset: {dataset}")
    print(args)
    data =  IGBHeteroDGLDatasetMassive(args)
    g = data.graph
    return g

def load_igbh_large():
    print("Loading igbh large dataset")
    return _load_igbh("large")

def load_igbh600m():
    return _load_igbh("full")


def load_reddit(self_loop=True):
    """Load reddit dataset."""
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes

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
    graph_data = {
        ('author', 'affiliated_with', 'institute'): (edge_index_affiliated_with[0,:], edge_index_affiliated_with[1,:]),
        ('paper', 'cites', 'paper'): (edge_index_cites[0,:], edge_index_cites[1,:]),
        ('author', 'writes', 'paper'): (edge_index_writes[ 0, :], edge_index_writes[ 1,:]),
    }

    num_nodes_dict = {'paper': dataset.num_papers, 'author': dataset.num_authors, 'institute': dataset.num_institutions}

    graph = dgl.heterograph(graph_data, num_nodes_dict)  

    return to_homogeneous(graph), dataset.num_classes

    

def load_ogb(name, root="dataset"):
    """Load ogbn dataset."""
    from ogb.nodeproppred import DglNodePropPredDataset
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
    assert is_pwd_correct_for_benchmark(), (
        "Please run this script at the repository root path."
        " The command will be something like python -m"
        " benchmark.partition_graph"
    )

    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="igbh600m",
        help="datasets: igbh600m, igbhlarge, reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=8, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="random", help="the partition method: random, metis"
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
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "igbh600m":
        g = load_igbh600m()
    elif args.dataset == "igbhlarge":
        g = load_igbh_large()
    elif args.dataset == "reddit":
        g, _ = load_reddit()
    elif args.dataset in ["ogbn-products", "ogbn-papers100M"]:
        g, _ = load_ogb(args.dataset)
    elif args.dataset in "mag240m":
        g, _ = load_ogb_lsc_mag_240m()
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    # Suppress the following print because it does not work for heterograph
    # print(
    #     "train: {}, valid: {}, test: {}".format(
    #         th.sum(g.ndata["train_mask"]),
    #         th.sum(g.ndata["val_mask"]),
    #         th.sum(g.ndata["test_mask"]),
    #     )
    # )
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    # dgl.distributed.partition_graph(
    #     g,
    #     args.dataset,
    #     args.num_parts,
    #     args.output,
    #     part_method=args.part_method,
    #     balance_ntypes=balance_ntypes,
    #     balance_edges=args.balance_edges,
    #     num_trainers_per_machine=args.num_trainers_per_machine,
    # )

    from .my_partition_graph import my_random_partition_graph
    my_random_partition_graph(g, args.dataset, args.num_parts, "out_data")