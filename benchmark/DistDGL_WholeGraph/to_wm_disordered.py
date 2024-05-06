"""This is an excerpt from the original DistDGL_WholeGraph/partition_graph.py that convert ordinary DGL graph partition to whole graph partition. The order of nodes were not preserved and the generated dataset is therefore only good for speed testing. In the original logic, 1) the wholegraph feature bin locations were added to the part_config file, and 2) the wholegraph feature bin files and json metadata were saved to the disk. We only do the 2) part in this script file, and modify the wholegraph loading logic to by default load the bin files and json metadata in the /wmg_features subdir in the same folder as the part_config file. Therefore, we only need to load the graph, and store the node feature of each node type in the /wmg_features subdir. We also extend the logic to support heterogeneous graph while the original logic only supports homogeneous graph."""

import os
import json
import argparse
import torch as th

from ..do_partition_graph import load_homogeneous_graph
from ..heterogeneous_version.do_partition_graph import load_heterogeneous_graph

if __name__ == "__main__":

    argparser = argparse.ArgumentParser("To WholeGraph Disordered")

    argparser.add_argument(
        "--output",
        type=str,
        default="out_igb240m_medium",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="igb240m_medium",
        help="datasets: igb240m_medium, ogbn-papers100M",
    )
    args = argparser.parse_args()

    wg_folder = os.path.join(args.output, "wg_features")

    try:
        g = load_homogeneous_graph(args.dataset)
        is_homogeneous = True
    except ValueError as e:
        print(
            "Failed to load homogeneous graph. Trying to load heterogeneous graph..."
        )
        g = load_heterogeneous_graph(args.dataset)
        is_homogeneous = False

    # Skip appending "wholegraph" for part_config

    # Load wholegraph feat
    if is_homogeneous:
        wg_feat = g.ndata["features"].to(dtype=th.float)
        wg_metadata = {}
        wg_metadata["features"] = {
            "shape": list(wg_feat.shape),
            "dtype": str(wg_feat.dtype),
        }

        with open(
            os.path.join(wg_folder, "{}_feat.json".format(args.dataset)), "w"
        ) as f:
            json.dump(wg_metadata, f)

        with open(
            os.path.join(wg_folder, "{}_feat.bin".format(args.dataset)), "wb"
        ) as f:
            print("Saving node feature to binary file...")
            wg_feat.numpy().tofile(f)
    else:
        # Store the node feature of each node type in the /wmg_features subdir
        raise NotImplementedError("Heterogeneous graph is not supported yet.")
