from dgl.random import choice as random_choice
from dgl.distributed.partition import _get_orig_ids, _get_inner_edge_mask, _get_inner_node_mask, _save_graphs, _dump_part_config
import os
from dgl.base import EID, ETYPE, NID, NTYPE
from dgl  import backend as F
import numpy as np

from dgl.data.utils import save_tensors
from dgl.partition import (
    get_peak_mem,
    partition_graph_with_halo,
)
import time
from dgl.distributed.graph_partition_book import (
    _etype_tuple_to_str,
)

"""This is a memory-efficient version of partition_graph() at github/dmlc/dgl/python/dgl/distributed/partition.py. We do not convert the input graph to a homogeneous graph."""
def my_random_partition_graph(g, 
    graph_name,
    num_parts,
    out_path,
    num_hops=1,num_trainers_per_machine=1,return_mapping=False, part_method="random",graph_formats=None):
    # sim_g is the converted homogeneous graph in the original code, we do not convert it here
    sim_g = g
    node_parts = random_choice(num_parts, sim_g.num_nodes())
    print("random_choice done", flush=True)
    if return_mapping:
        sim_g.ndata["orig_id"] = F.arange(0, sim_g.num_nodes())
        sim_g.edata["orig_id"] = F.arange(0, sim_g.num_edges())
    # KWU: Convert sim_g[ETYPE] to int64
    sim_g.edata[ETYPE] = F.astype(sim_g.edata[ETYPE], F.int64)
    # KWU: Reshuffle is a must because when the dist trainer loaded the graph, it assmes the indices in each parts are contiguous as a result of reshuffling
    parts, orig_nids, orig_eids = partition_graph_with_halo(
        sim_g, node_parts, num_hops, reshuffle=True
    )
    print("partition_graph_with_halo done", flush=True)
    # Node mapping is in halo_subg.induced_vertices
    # according to GetSubgraphWithHalo dmlc/dgl/src/graph/transform/partition_hetero.cc
    # It can be obtained in python by subg.induced_nodes()
    
    if return_mapping:
        raise NotImplementedError("return_mapping needs to be rewritten")
        orig_nids, orig_eids = _get_orig_ids(g, sim_g, orig_nids, orig_eids)
    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # With reshuffling, we can ensure that all nodes and edges are reshuffled
    # and are in contiguous ID space.
    if num_parts > 1:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            val = []
            node_map_val[ntype] = []
            for i in parts:
                inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                val.append(
                    F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0))
                )
                inner_nids = F.boolean_mask(
                    parts[i].ndata[NID], inner_node_mask
                )
                node_map_val[ntype].append(
                    [
                        int(F.as_scalar(inner_nids[0])),
                        int(F.as_scalar(inner_nids[-1])) + 1,
                    ]
                )
            val = np.cumsum(val).tolist()
            assert val[-1] == g.num_nodes(ntype)
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            val = []
            edge_map_val[etype] = []
            for i in parts:
                inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                val.append(
                    F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0))
                )
                inner_eids = np.sort(
                    F.asnumpy(
                        F.boolean_mask(parts[i].edata[EID], inner_edge_mask)
                    )
                )
                edge_map_val[etype].append(
                    [int(inner_eids[0]), int(inner_eids[-1]) + 1]
                )
            val = np.cumsum(val).tolist()
            assert val[-1] == g.num_edges(etype)
    else:
        raise NotImplementedError("num_parts == 1 is not supported yet")
    
    print("map val production done", flush=True)
    
    start = time.time()
    ntypes = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype: g.get_etype_id(etype) for etype in g.canonical_etypes}
    part_metadata = {
        "graph_name": graph_name,
        "num_nodes": g.num_nodes(),
        "num_edges": g.num_edges(),
        "part_method": part_method,
        "num_parts": num_parts,
        "halo_hops": num_hops,
        "node_map": node_map_val,
        "edge_map": edge_map_val,
        "ntypes": ntypes,
        "etypes": etypes,
    }
    for part_id in range(num_parts):
        part = parts[part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                # To get the edges in the input graph, we should use original node IDs.
                # Both orig_id and NID stores the per-node-type IDs.
                # KWU: If reshuffling is not done, use instead ndata_name = NID. However, later loading will trigger error because the indices in each part is no longer contiguous.
                ndata_name = "orig_id"
                inner_node_mask = _get_inner_node_mask(part, ntype_id)
                # This is global node IDs.
                local_nodes = F.boolean_mask(
                    part.ndata[ndata_name], inner_node_mask
                )
                if len(g.ntypes) > 1:
                    # If the input is a heterogeneous graph.
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                    print(
                        "part {} has {} nodes of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.ndata[NTYPE] == ntype_id, 0)
                            ),
                            ntype,
                            len(local_nodes),
                        ), flush=True
                    )
                else:
                    print(
                        "part {} has {} nodes and {} are inside the partition".format(
                            part_id, part.num_nodes(), len(local_nodes)
                        ), flush=True
                    )

                for name in g.nodes[ntype].data:
                    if name in [NID, "inner_node"]:
                        continue
                    node_feats[ntype + "/" + name] = F.gather_row(
                        g.nodes[ntype].data[name], local_nodes
                    )

            for etype in g.canonical_etypes:
                etype_id = g.get_etype_id(etype)
                # KWU: If reshuffling is not done, use instead edata_name = EID. However, later loading will trigger error because the indices in each part is no longer contiguous.
                edata_name = "orig_id"
                inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                # This is global edge IDs.
                local_edges = F.boolean_mask(
                    part.edata[edata_name], inner_edge_mask
                )
                if not g.is_homogeneous:
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                    print(
                        "part {} has {} edges of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.edata[ETYPE] == etype_id, 0)
                            ),
                            etype,
                            len(local_edges),
                        ), flush=True
                    )
                else:
                    print(
                        "part {} has {} edges and {} are inside the partition".format(
                            part_id, part.num_edges(), len(local_edges)
                        ), flush=True
                    )
                tot_num_inner_edges += len(local_edges)

                for name in g.edges[etype].data:
                    if name in [EID, "inner_edge"]:
                        continue
                    edge_feats[
                        _etype_tuple_to_str(etype) + "/" + name
                    ] = F.gather_row(g.edges[etype].data[name], local_edges)

        else:
            raise NotImplementedError("num_parts == 1 is not supported yet")

        if return_mapping:
            # delete `orig_id` from ndata/edata
            del part.ndata["orig_id"]
            del part.edata["orig_id"]

        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata["part-{}".format(part_id)] = {
            "node_feats": os.path.relpath(node_feat_file, out_path),
            "edge_feats": os.path.relpath(edge_feat_file, out_path),
            "part_graph": os.path.relpath(part_graph_file, out_path),
        }
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        sort_etypes = len(g.etypes) > 1
        _save_graphs(
            part_graph_file,
            [part],
            formats=graph_formats,
            sort_etypes=sort_etypes,
        )
    print(
        "Save partitions: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        ), flush=True
    )

    _dump_part_config(f"{out_path}/{graph_name}.json", part_metadata)

    num_cuts = sim_g.num_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print(
        "There are {} edges in the graph and {} edge cuts for {} partitions.".format(
            g.num_edges(), num_cuts, num_parts
        ), flush=True
    )

    if return_mapping:
        return orig_nids, orig_eids