# From https://github.com/dmlc/dgl/blob/master/examples/distributed/graphsage/partition_graph.py
import argparse
import time

import dgl
import torch as th
import time

from dgl.convert import to_homogeneous
from .utils import (
    get_igb_config,
    is_pwd_correct_for_benchmark,
    construct_graph_attributes,
)
from igb.dataloader import IGB260MDGLDataset


def _load_igb(dataset_size: str, use_dummy_feats: bool = True):
    args = get_igb_config()
    if dataset_size == "medium":
        args.dataset_size = "medium"
    elif dataset_size == "large":
        args.dataset_size = "large"
    elif dataset_size == "full":
        pass
    else:
        raise ValueError(f"Unknown igb dataset_size: {dataset_size}")
    if not use_dummy_feats:
        args.dummy_feats = 0
    print(args, flush=True)
    if args.dummy_feats:
        print("using dummy feats")
    data = IGB260MDGLDataset(args)
    g = data.graph
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g


def load_igb240m():
    return _load_igb("full")


def load_igb240m_medium():
    return _load_igb("medium", use_dummy_feats=False)


def load_reddit(self_loop=True):
    """Load reddit dataset."""
    from dgl.data import RedditDataset

    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


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


def load_homogeneous_graph(dataset: str):
    if (
        dataset == "igbh600m"
        or dataset == "igbhmedium"
        or dataset == "igbhlarge"
        or dataset == "mag240m"
    ):
        raise ValueError(
            "Please use the graph partitioning and distributed training script in .heterogeneous_version"
        )
    elif dataset == "igb240m":
        g = load_igb240m()
    elif dataset == "igb240m_medium":
        g = load_igb240m_medium()
    elif dataset == "reddit":
        g, _ = load_reddit()
    elif dataset in ["ogbn-products", "ogbn-papers100M"]:
        g, _ = load_ogb(dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return g


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
        help="datasets: igbh600m, igbhlarge, igb240m, igb240m_medium, reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--num_parts",
        type=int,
        default=2,
        help="number of partitions (i.e. compute nodes in training)",
    )
    argparser.add_argument(
        "--part_method",
        type=str,
        default="random",
        help="the partition method: random, metis",
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
        "--heterogeneous",
        action="store_true",
        help="not convert the graph to a homogeneous graph.",
    )
    argparser.add_argument(
        "--memory_efficient_impl",
        action="store_true",
        help="Use a slightly more memory-efficient implementation to avoid oom.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=2,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    args = argparser.parse_args()
    if args.heterogeneous:
        raise NotImplementedError(
            "Please use the graph partitioning and distributed training script in .heterogeneous_version"
        )

    start = time.time()

    g = load_homogeneous_graph(args.dataset)

    print(
        "Load {} takes {:.3f} seconds".format(
            args.dataset, time.time() - start
        ),
        flush=True,
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()), flush=True)
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

    start = time.time()
    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
    print(
        "Converted to (or skipped the conversion of) undirected graph {:.3f}".format(
            time.time() - start
        ),
        flush=True,
    )

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

    if args.memory_efficient_impl:
        if args.heterogeneous:
            # g_attrs needs to be passed to partition_graph. In case of heterogeneous graph, it should be from the original heterogeneous graph before conversion to homogeneous graph.
            g_attrs = construct_graph_attributes(g)
        g = dgl.to_homogeneous(g)
        if not args.heterogeneous:
            # g_attrs needs to be passed to partition_graph. In case of homogeneous graph, it should be from the homogeneous graph.
            g_attrs = construct_graph_attributes(g)
        print("Converted to homogeneous graph", flush=True)
        from .my_partition_graph import my_random_partition_graph

        my_random_partition_graph(
            g, g_attrs, args.dataset, args.num_parts, "out_single_dataset"
        )
    else:
        from dgl.distributed.partition import partition_graph

        start = time.time()
        partition_graph(
            g,
            args.dataset,
            args.num_parts,
            f"out_{args.dataset}_{args.num_parts}_{args.num_trainers_per_machine}_with_metis",
            #part_method="random",
            part_method="metis",
        )
        print(
            "Partitioning graph takes {:.3f} seconds".format(
                time.time() - start
            ),
            flush=True,
        )
