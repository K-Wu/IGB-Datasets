from __future__ import annotations
from typing import Any
import numpy as np
from dgl  import backend as F
from dgl.convert import graph
import torch as th
from dgl.base import EID, ETYPE, NID, NTYPE

def get_unique_temp_file_name():
    import tempfile
    import os
    return os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))

def cat_mmap_arrays(arrays, out=None, mode="r"):
    """Concatenate arrays and save it in mmaped storage.
    Parameters
    ----------
    arrays : list of arrays
        Arrays to concatenate.
    out : mmap array, optional
        The output array. If not given, a new array is created.
    mode : str, optional
        The mode of the output array. Can be 'r' or 'w'.
    Returns
    -------
    mmap array
        The concatenated array.
    """
    if out is None:
        out = th.IntTensor(th.IntStorage.from_file(get_unique_temp_file_name(), True, size=sum(a.shape[0] for a in arrays)))
    offset = 0
    for a in arrays:
        out[offset : offset + a.shape[0]] = a
        offset += a.shape[0]
    return out

def construct_homogeneous(
    G_all_edges:dict[tuple[str,str,str],tuple[Any, Any]],G_canonical_etypes,
    G_ntypes, G_ntype_to_num_nodes: dict[str, int],store_type=True, return_count=False,G_idtype=F.int32,G_device="cpu"
    
):
    """A simplified version based on to_homogeneous in dmlc/dgl/python/dgl/convert.py.
    It is used to create to_homogeneous from scratch according to the data that creates the heterogeneous graph as a memory-efficient implementation.
    Some arguments in to_homogeneous are not supported."""
    G_ntype_id = {ntype: i for i, ntype in enumerate(G_ntypes)}

    num_nodes_per_ntype = [G_ntype_to_num_nodes[ntype] for ntype in G_ntypes]
    offset_per_ntype = np.insert(np.cumsum(num_nodes_per_ntype), 0, 0)
    srcs = []
    dsts = []
    nids = []
    eids = []
    if store_type:
        ntype_ids = []
        etype_ids = []
    if return_count:
        ntype_count = []
        etype_count = []
    total_num_nodes = 0

    for ntype_id, ntype in enumerate(G_ntypes):
        num_nodes = G_ntype_to_num_nodes[ntype]
        total_num_nodes += num_nodes
        if store_type:
            # Type ID is always in int64
            ntype_ids.append(F.full_1d(num_nodes, ntype_id, F.int64, G_device))
        if return_count:
            ntype_count.append(num_nodes)
        nids.append(F.arange(0, num_nodes, G_idtype, G_device))

    for etype_id, etype in enumerate(G_canonical_etypes):
        srctype, _, dsttype = etype
        src, dst = G_all_edges[etype]
        num_edges = len(src)
        srcs.append(src + int(offset_per_ntype[G_ntype_id[srctype]]))
        dsts.append(dst + int(offset_per_ntype[G_ntype_id[dsttype]]))
        if store_type:
            # Type ID is always in int64
            etype_ids.append(F.full_1d(num_edges, etype_id, F.int64, G_device))
        if return_count:
            etype_count.append(num_edges)
        eids.append(F.arange(0, num_edges, G_idtype, G_device))

    retg = graph(
        #(F.cat(srcs, 0), F.cat(dsts, 0)),
        (cat_mmap_arrays(srcs), cat_mmap_arrays(dsts)),
        num_nodes=total_num_nodes,
        idtype=G_idtype,
        device=G_device,
    )

    # # copy features
    # #if ndata is None:
    # ndata = []
    # #if edata is None:
    # edata = []
    # comb_nf = combine_frames(
    #     G._node_frames, range(len(G_ntypes)), col_names=ndata
    # )
    # comb_ef = combine_frames(
    #     G._edge_frames, range(len(G_etypes)), col_names=edata
    # )
    # if comb_nf is not None:
    #     retg.ndata.update(comb_nf)
    # if comb_ef is not None:
    #     retg.edata.update(comb_ef)

    retg.ndata[NID] = cat_mmap_arrays(nids)
    retg.edata[EID] = cat_mmap_arrays(eids)
    if store_type:
        retg.ndata[NTYPE] = cat_mmap_arrays(ntype_ids)
        retg.edata[ETYPE] = cat_mmap_arrays(etype_ids)

    if return_count:
        return retg, ntype_count, etype_count
    else:
        return retg