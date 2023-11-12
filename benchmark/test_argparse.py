import argparse
from .utils import is_pwd_correct_for_benchmark

if __name__ == "__main__":
    assert is_pwd_correct_for_benchmark(), (
          "Please run this script at the repository root path."
          " The command will be something like python -m"
          " benchmark.test_argparse"
      )
    # This import should work if pwd is the repository root path
    import igb.dataloader
    argparser = argparse.ArgumentParser("Partition graph")
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
    args = argparser.parse_args()
    print(args)