# From https://github.com/dmlc/dgl/blob/master/examples/distributed/graphsage/node_classification.py

import argparse
import socket
import time

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.nn.pytorch import GATConv, HeteroGraphConv
from dgl import apply_each
import os

class DistRGAT(nn.Module):
    """Adapted from class GAT in /IGB-datasets/igb/models.py. Arguments of __init__ are renamed to unify with the DistSAGE class."""
    def __init__(self, etypes, in_feats, n_hidden, n_classes, n_heads, n_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, n_hidden // n_heads, n_heads)
            for etype in etypes}))
        for _ in range(n_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(n_hidden, n_hidden // n_heads, n_heads)
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(n_hidden, n_hidden // n_heads, n_heads)
            for etype in etypes}))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])   

      

class DistGAT(nn.Module):
    """Adapted from class GAT in /IGB-datasets/igb/models.py. Arguments of __init__ are renamed to unify with the DistSAGE class."""
    def __init__(self, in_feats, n_hidden, n_classes, n_heads, n_layers=2, dropout=0.2):
        super(DistGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden, n_heads, allow_zero_in_degree=True))
        for _ in range(n_layers-2):
            self.layers.append(GATConv(n_hidden * n_heads, n_hidden, n_heads, allow_zero_in_degree=True))
        self.layers.append(GATConv(n_hidden * n_heads, n_classes, n_heads, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            if l < len(self.layers) - 1:
                h = layer(block, (h, h_dst)).flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = layer(block, (h, h_dst)).mean(1)  
        return h

class DistSAGE(nn.Module):
    """
    SAGE model for distributed train and evaluation.

    Parameters
    ----------
    in_feats : int
        Feature dimension.
    n_hidden : int
        Hidden layer dimension.
    n_classes : int
        Number of classes.
    n_layers : int
        Number of layers.
    activation : callable
        Activation function.
    dropout : float
        Dropout value.
    """

    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        """
        Forward function.

        Parameters
        ----------
        blocks : List[DGLBlock]
            Sampled blocks.
        x : DistTensor
            Feature data.
        """
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Distributed layer-wise inference with the GraphSAGE model on full
        neighbors.

        Parameters
        ----------
        g : DistGraph
            Input Graph for inference.
        x : DistTensor
            Node feature data of input graph.

        Returns
        -------
        DistTensor
            Inference results.
        """
        # Split nodes to each trainer.
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        for i, layer in enumerate(self.layers):
            # Create DistTensor to save forward results.
            if i == len(self.layers) - 1:
                out_dim = self.n_classes
                name = "h_last"
            else:
                out_dim = self.n_hidden
                name = "h"
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), out_dim),
                th.float32,
                name,
                persistent=True,
            )
            print(f"|V|={g.num_nodes()}, inference batch size: {batch_size}", flush=True)

            # `-1` indicates all inbound edges will be inlcuded, namely, full
            # neighbor sampling.
            sampler = dgl.dataloading.NeighborSampler([-1], fused=False) # fused=True is not supported in DistGraph
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # Copy back to CPU as DistTensor requires data reside on CPU.
                y[output_nodes] = h.cpu()

            x = y
            # Synchronize trainers.
            g.barrier()
        return x


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    labels : torch.Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Accuracy.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation and test set.

    Parameters
    ----------
    model : DistSAGE
        The model to be evaluated.
    g : DistGraph
        The entire graph.
    inputs : DistTensor
        The feature data of all the nodes.
    labels : DistTensor
        The labels of all the nodes.
    val_nid : torch.Tensor
        The node IDs for validation.
    test_nid : torch.Tensor
        The node IDs for test.
    batch_size : int
        Batch size for evaluation.
    device : torch.Device
        The target device to evaluate on.

    Returns
    -------
    float
        Validation accuracy.
    float
        Test accuracy.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, data):
    """
    Train and evaluate DistSAGE.
    The training uses the logic for heterogeneous GNNs when args.heterogeneous is set: in this case, the "paper" type nodes are to be predicted.

    Parameters
    ----------
    args : argparse.Args
        Arguments for train and evaluate.
    device : torch.Device
        Target device for train and evaluate.
    data : Packed Data
        Packed data includes train/val/test IDs, feature dimension,
        number of classes, graph.
    """
    host_name = socket.gethostname()
    if args.use_wm:
        train_nid, val_nid, test_nid, in_feats, n_classes, g, wm_features = data
        if args.heterogeneous:
            wm_features, num_features_offset = wm_features
    else:
        train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    if args.heterogeneous:
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            {"paper": train_nid},
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    else:
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    if args.model == "DistSAGE":
        assert not args.heterogeneous
        model = DistSAGE(
            in_feats,
            args.num_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
        )
    elif args.model == "DistGAT":
        print(f"{host_name} {g.rank()}: Using DistGAT model default parameters batch_size(2048) fanout (5,2,2,2) n_layers (4), hidden_dim 512, n_heads (4)", flush=True)
        assert not args.heterogeneous
        assert args.batch_size == 2048
        assert args.fan_out == "5,2,2,2" or args.fan_out == "10,5,5"
        assert args.n_layers == 4 or args.n_layers == 3
        assert args.num_hidden == 512
        model = DistGAT(
            in_feats,
            n_hidden = args.num_hidden,
            n_classes = n_classes,
            n_heads = 4,
            n_layers = args.n_layers,
            dropout = args.dropout,
        )
    elif args.model == "DistRGAT":
        print(f"{host_name} {g.rank()}: Using DistRGAT model default parameters batch_size(2048) fanout (5,2,2,2) n_layers (4), hidden_dim 512, n_heads (4)", flush=True)
        assert args.heterogeneous
        assert args.batch_size == 2048
        assert args.fan_out == "5,2,2,2" or args.fan_out == "10,5,5"
        assert args.n_layers == 4 or args.n_layers == 3
        if args.graph_name!="mag240m":
            assert args.num_hidden == 512
        model = DistRGAT(
            g.etypes,
            in_feats,
            n_hidden = args.num_hidden,
            n_classes = n_classes,
            n_heads = 4,
            n_layers = args.n_layers,
            dropout = args.dropout,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    model = model.to(device)
    if args.num_gpus == 0:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop.
    iter_tput = []
    epoch = 0
    epoch_time = []
    test_acc = 0.0
    for _ in range(args.num_epochs):
        epoch += 1
        tic = time.time()
        # Various time statistics.
        sample_and_aggregate_times = []
        forward_times = []
        backward_times = []
        update_times = []
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        step_time = []
        sampled_times_sampling = []
        sampled_times_movement = []
        sampled_times_training = []
        sampled_step_beg =  1 # 500
        assert sampled_step_beg >=1, "sampled_step_beg must be at least 1 because we need to execute dataloader.set_print_times(g.rank()) in the previous step"
        sampled_step_end = 101 # 600
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()
                sample_and_aggregate_times.append(tic_step - start)  # KWU: Sample and aggregation time
                # Slice feature and label.
                if args.heterogeneous:
                    # print("blocks", blocks, flush=True)
                    # print("blocks[-1]", blocks[-1], flush=True)
                    # print("blocks[0]", blocks[0], flush=True)
                    # print("blocks[-1].dstdata", blocks[-1].dstdata, flush=True)
                    # print("blocks[0].srcdata", blocks[0].srcdata, flush=True)
                    # print("input_ndoes", input_nodes, flush=True)
                    # print("seeds", seeds, flush=True)
                    # batch_labels = blocks[-1].dstdata['labels']#['paper']
                    # seeds = seeds["paper"]
                    # TODO: add wholegraph support according to L144 in benchmark/DistDGL_WholeGraph/node_classification.py
                    if args.use_wm:
                        raise NotImplementedError
                    else:
                        batch_inputs = {ntype: g.nodes[ntype].data["features"][input_nodes[ntype]] for ntype in g.ntypes}
                    batch_labels = g.nodes["paper"].data["labels"][seeds['paper']].long()
                    # number_train += seeds["paper"].shape[0]
                    num_inputs += np.sum(
                    [blocks[0].num_src_nodes(ntype) for ntype in blocks[0].ntypes]
                    )
                    num_seeds = np.sum([blocks[-1].num_dst_nodes(ntype) for ntype in blocks[-1].ntypes])
                else:
                    # TODO: add wholegraph support according to L144 in benchmark/DistDGL_WholeGraph/node_classification.py
                    if args.use_wm:
                        batch_inputs = wm_features.gather(input_nodes.cuda())
                    else:
                        batch_inputs = g.ndata["features"][input_nodes]
                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])
                    batch_labels = g.ndata["labels"][seeds].long()
                
                # Move to target device.
                blocks = [block.to(device) for block in blocks]
                if args.heterogeneous:
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    #batch_labels = {k: v.to(device) for k, v in batch_labels.items()}
                    batch_labels = batch_labels.to(device)
                    # print("batch_inputs", batch_inputs, flush=True)
                    # print("batch_labels", batch_labels, flush=True)
                else:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)
                # Compute loss and prediction.
                start = time.time()
                movement_time = start - tic_step # KWU: Movement time
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_times.append(forward_end - start)
                backward_times.append(compute_end - forward_end)

                optimizer.step()
                update_times.append(time.time() - compute_end)
                train_time = time.time() - start # KWU: Train time

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)

                if step >= sampled_step_beg and step < sampled_step_end:
                    sampled_times_sampling.append(sample_and_aggregate_times[-1])
                    sampled_times_movement.append(movement_time)
                    sampled_times_training.append(train_time)
                    print(f"{host_name} {g.rank()}: Part {g.rank()} | Epoch {epoch:05d} | Step {step:05d} | Sample + Aggregation Time {sample_and_aggregate_times[-1]:.4f} sec | Movement Time {movement_time:.4f} sec | Train Time {train_time:.4f} sec", flush=True)
                
                if step == sampled_step_beg - 1:
                    sampler.set_print_times()
                    dataloader.set_print_times(g.rank())
                if step == sampled_step_end:
                    # Exit the loop
                    sampler.reset_print_times()
                    dataloader.reset_print_times()
                    break
                # Print stats every args.log_every steps.
                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    sample_speed = np.mean(iter_tput[-args.log_every :])
                    mean_step_time = np.mean(step_time[-args.log_every :])
                    print(
                        f"{host_name} {g.rank()}: Part {g.rank()} | Epoch {epoch:05d} | Step {step:05d}"
                        f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                        f" | Speed (samples/sec) {sample_speed:.4f}"
                        f" | GPU {gpu_mem_alloc:.1f} MB | "
                        f"Mean step time {mean_step_time:.3f} s", flush=True
                    )
                start = time.time()

        toc = time.time()
        print(
            f"{host_name} {g.rank()}: Part {g.rank()}, Epoch Time(s): {toc - tic:.4f}, "
            f"sample+data_copy: {sum(sample_and_aggregate_times)/len(sample_and_aggregate_times):.4f}, forward: {sum(forward_times)/len(forward_times):.4f},"
            f" backward: {sum(backward_times)/len(backward_times):.4f}, update: {sum(update_times)/len(update_times):.4f}, "
            f"#seeds: {num_seeds}, #inputs: {num_inputs}", flush=True
        )
        epoch_time.append(toc - tic)

        # TODO: work on DistGAT.inference()
        # TODO: work on wholgraph
        if not (args.heterogeneous or args.use_wm):
            if (epoch % args.eval_every == 0 or epoch == args.num_epochs) and isinstance(model, DistGAT):
                start = time.time()
                val_acc, test_acc = evaluate(
                    model.module,
                    g,
                    g.ndata["features"],
                    g.ndata["labels"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                )
                print(
                    f"Part {g.rank()}, Val Acc {val_acc:.4f}, "
                    f"Test Acc {test_acc:.4f}, time: {time.time() - start:.4f}", flush=True
                )

    return np.mean(epoch_time[-int(args.num_epochs * 0.8) :]), test_acc


def main(args):
    """
    Main function.
    """
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.", flush=True)
    dgl.distributed.initialize(args.ip_config)
    print(f"{host_name}: Initializing PyTorch process group.", flush=True)
    th.distributed.init_process_group(backend=args.backend)
    print(f"{host_name}: Initializing DistGraph.", flush=True)
    # e.g., g = dgl.distributed.DistGraph("igbh",part_config="./out_data_2_2/igbh600m.json")
    # Either enable shared memory of the server (by default by dgl), or specify gpb (partition book) in the following DistGraph initiation argument.
    if args.use_wm:
        assert args.part_config.endswith("_with_wg.json"), "part_config must ends with '_with_wg.json'"
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(f"{host_name} {g.rank()}: Rank of {host_name}: {g.rank()}", flush=True)

    if args.regenerate_node_features:
        print(f"{host_name} {g.rank()}: Regenerating node features.", flush=True)
        print(f"{host_name} {g.rank()}: num_nodes ", g.num_nodes(), flush=True)
        g.ndata["features"] = th.randn(g.num_nodes(), 1024)

    # Split train/val/test IDs for each trainer.
    pb = g.get_partition_book()
    if args.heterogeneous:
        if "trainer_id" in g.nodes["paper"].data:
            train_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["train_mask"],
                pb,
                ntype="paper",
                force_even=True,
                node_trainer_ids=g.nodes["paper"].data["trainer_id"],
            )
            val_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["val_mask"],
                pb,
                ntype="paper",
                force_even=True,
                node_trainer_ids=g.nodes["paper"].data["trainer_id"],
            )
            test_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["test_mask"],
                pb,
                ntype="paper",
                force_even=True,
                node_trainer_ids=g.nodes["paper"].data["trainer_id"],
            )
        else:
            train_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["train_mask"],
                pb,
                ntype="paper",
                force_even=True,
            )
            val_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["val_mask"],
                pb,
                ntype="paper",
                force_even=True,
            )
            test_nid = dgl.distributed.node_split(
                g.nodes["paper"].data["test_mask"],
                pb,
                ntype="paper",
                force_even=True,
            )
    else:
        if "trainer_id" in g.ndata:
            train_nid = dgl.distributed.node_split(
                g.ndata["train_mask"],
                pb,
                force_even=True,
                node_trainer_ids=g.ndata["trainer_id"],
            )
            val_nid = dgl.distributed.node_split(
                g.ndata["val_mask"],\
                pb,
                force_even=True,
                node_trainer_ids=g.ndata["trainer_id"],
            )
            test_nid = dgl.distributed.node_split(
                g.ndata["test_mask"],
                pb,
                force_even=True,
                node_trainer_ids=g.ndata["trainer_id"],
            )
            local_nid = pb.partid2nids(pb.partid, "paper").detach().numpy()
            print(
                "part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})".format(
                    g.rank(),
                    len(train_nid),
                    len(np.intersect1d(train_nid.numpy(), local_nid)),
                    len(val_nid),
                    len(np.intersect1d(val_nid.numpy(), local_nid)),
                    len(test_nid),
                    len(np.intersect1d(test_nid.numpy(), local_nid)),
                )
            )
        else:
            train_nid = dgl.distributed.node_split(
                g.ndata["train_mask"], pb, force_even=True
            )
            val_nid = dgl.distributed.node_split(
                g.ndata["val_mask"], pb, force_even=True
            )
            test_nid = dgl.distributed.node_split(
                g.ndata["test_mask"], pb, force_even=True
            )
            local_nid = pb.partid2nids(pb.partid).detach().numpy()
            num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid))
            num_val_local = len(np.intersect1d(val_nid.numpy(), local_nid))
            num_test_local = len(np.intersect1d(test_nid.numpy(), local_nid))
            print(
                f"{host_name} {g.rank()}: part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
                f"val: {len(val_nid)} (local: {num_val_local}), "
                f"test: {len(test_nid)} (local: {num_test_local})", flush=True
            )
        del local_nid
    if args.num_gpus <= 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    # Set default cuda device for the wholegraph communicator
    th.cuda.set_device(device)
    n_classes = args.n_classes
    if n_classes == 0:
        if args.heterogeneous:
            labels = g.nodes["paper"].data["labels"][np.arange(g.num_nodes("paper"))]
        else:
            labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print(f"{host_name} {g.rank()}: Number of classes: {n_classes}", flush=True)

    if args.use_wm:
        # init and load features into wholegraph feature store
        # Add this to args to pass to init_wholegraph and following wholegraph APIs
        args.ngpu_per_node = args.num_gpus 
        args.dataset = os.path.basename(args.part_config)[:-len("_with_wg.json")]
        feat_comm = init_wholegraph(args)
        dev_id = th.cuda.current_device()
        config_path = os.environ.get("DGL_CONF_PATH")
        feat_dim, wg_path = parse_wholegraph_config(config_path, args.dataset)
        # Pack data
        if args.heterogeneous:
            raise NotImplementedError
            node_feat_wm_embedding, num_features_offset = load_wholegraph_distribute_feature_tensor(
                feat_comm, feat_dim, feat_path, args.dataset, args.wm_feat_location
            )
            in_feats = num_features_offset["paper"][1] - num_features_offset["paper"][0]
            wm_features = (node_feat_wm_embedding, num_features_offset)
        else:
            node_feat_wm_embedding = load_wholegraph_distribute_feature_tensor(
                feat_comm, feat_dim, wg_path, args.dataset, args.wm_feat_location
            )
            in_feats = node_feat_wm_embedding.shape[1]
            wm_features = node_feat_wm_embedding
        data = train_nid, val_nid, test_nid, in_feats, n_classes, g, wm_features
    else:
        # Pack data.
        if args.heterogeneous:
            in_feats = g.nodes["paper"].data['features'][np.arange(g.num_nodes("paper"))].shape[1]
        else:
            in_feats = g.ndata["features"].shape[1]
        data = train_nid, val_nid, test_nid, in_feats, n_classes, g


    # Train and evaluate.
    epoch_time, test_acc = run(args, device, data)
    print(
        f"{host_name} {g.rank()}: Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f}", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--model", type=str, default="DistSAGE", help="model to use. DistSAGE or DistGAT"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="the number of GPU device. Use 0 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument(
        "--regenerate_node_features",
        default=False,
        action="store_true",
        help="Regenerate node features.",
    )
    
    parser.add_argument(
        "--wg-launch-agent", type=str, choices=["pytorch", "mpi"], default="pytorch",
        help="Initialize wholegraph communication backend through pytorch, or mpi (srun)"
    )
    parser.add_argument(
        "--wg-comm-backend", type=str, choices=["nccl", "nvshmem"], default="nccl",
        help="WholeGraph communication backend library using nccl or nvshmem"
    )
    parser.add_argument(
        "--use-wm",
        action="store_true",
        help="turn the features into wholegraph compatible format.",
    )
    parser.add_argument(
        "--wm-feat-location",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="feature store at [cpu|cuda]",
    )

    args = parser.parse_args()
    
    if args.wg_comm_backend == 'nvshmem':
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "15g"

    if args.use_wm:
        import torch.distributed as dist
        from .DistDGL_WholeGraph.utils.wholegraph_launch import (
            init_wholegraph,
            parse_wholegraph_config,
            create_wholegraph_dist_tensor,
            is_wm_tensor,
            wm_scatter
        )
        from .DistDGL_WholeGraph.utils.load_feature import (
            load_wholegraph_distribute_feature_tensor,
        )

    print(f"Arguments: {args}", flush=True)
    main(args)