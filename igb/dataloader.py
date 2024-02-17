import argparse
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")

def is_this_machine_bafs():
    import socket
    return socket.gethostname() == 'bafs-01'

BAFS_FILE_MAPPING = {
'//full/processed/paper/node_label_19.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy',
'//full/processed/paper/node_label_2K.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy',
'//full/processed/paper/paper_id_index_mapping.npy': '/mnt/nvme16/IGB260M_part_2/processed/paper/paper_id_index_mapping.npy',
'//full/processed/paper__cites__paper/edge_index.npy': '/mnt/nvme16/IGB260M_part_2/processed/paper__cites__paper/edge_index.npy',
'//full/processed/paper/node_feat.npy': '/mnt/nvme17/node_feat.npy',
'//full/processed/fos/fos_id_index_mapping.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/fos/fos_id_index_mapping.npy',
'//full/processed/fos/node_feat.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/fos/node_feat.npy',
'//full/processed/paper__topic__fos/edge_index.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__topic__fos/edge_index.npy',
'//full/processed/paper/paper_id_index_mapping.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper/paper_id_index_mapping.npy',
'//full/processed/paper__cites__paper/edge_index_col_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_col_idx.npy',
'//full/processed/paper__cites__paper/edge_index_csc_edge_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_edge_idx.npy',
'//full/processed/paper__cites__paper/edge_index_edge_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_edge_idx.npy',
'//full/processed/paper__cites__paper/edge_index_row_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_row_idx.npy',
'//full/processed/paper__cites__paper/edge_index_csc_col_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_col_idx.npy',
'//full/processed/paper__cites__paper/edge_index_csc_row_idx.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_row_idx.npy',
'//full/processed/paper__cites__paper/edge_index.npy': '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
}

def get_bafs_path(*relative_path_args):
    if '//' + osp.join(*relative_path_args) in BAFS_FILE_MAPPING:
        return BAFS_FILE_MAPPING['//' + osp.join(*relative_path_args)]
    else:
        relative_path_args = list(relative_path_args)
        relative_path_args = ['mnt', 'raid0'] + relative_path_args
        return '/' + osp.join(*relative_path_args)
    


# Overwritten by the _extended version. aws s3 cp /mnt/nvme16/IGB260M_part_2/processed/paper/node_label_19.npy s3://vikram3/full/processed/paper/node_label_19.npy
# Overwritten by the _extended version. aws s3 cp /mnt/nvme16/IGB260M_part_2/processed/paper/node_label_2K.npy s3://vikram3/full/processed/paper/node_label_2K.npy
# Overwritten by the repeated command. aws s3 cp /mnt/nvme17/node_feat.npy s3://vikram3/full/processed/paper/node_feat.npy --cli-connect-timeout 3600

class IGB260M(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int, dummy_feats: bool):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes
        self.dummy_feats = dummy_feats

    def num_nodes(self):
        if self.size == 'experimental':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        if self.dummy_feats:
            return np.ones((num_nodes, 1))
        # TODO: temp for bafs. large and full special case
        if self.size == 'large' or self.size == 'full':
            if is_this_machine_bafs():
                path = get_bafs_path('full', 'processed', 'paper', 'node_feat.npy')
            else:
                path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_feat.npy')
            emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            if is_this_machine_bafs():
                path = get_bafs_path(self.size, 'processed', 'paper', 'node_feat.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:

        if self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                if is_this_machine_bafs():
                    path = get_bafs_path('full', 'processed', 'paper', 'node_label_19.npy')
                else:
                    path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                if is_this_machine_bafs():
                    path = get_bafs_path('full', 'processed', 'paper', 'node_label_2K.npy')
                else:
                    path = osp.join(self.dir, 'full', 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                if is_this_machine_bafs():
                    path = get_bafs_path(self.size, 'processed', 'paper', 'node_label_19.npy')
                else:
                    path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                if is_this_machine_bafs():
                    path = get_bafs_path(self.size, 'processed', 'paper', 'node_label_2K.npy')
                else:
                    path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        if is_this_machine_bafs():
            path = get_bafs_path(self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        # if self.size == 'full':
        #     path = '/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')


class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260MDGLDataset')

    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, \
            classes=self.args.num_classes, synthetic=self.args.synthetic,dummy_feats=self.args.dummy_feats)

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val   = int(n_labeled_idx * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        if self.args.in_memory:
            if is_this_machine_bafs():
                paper_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            else:   
                paper_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            paper_paper_edges = torch.from_numpy(np.load(paper_paper_edges_path))

            if is_this_machine_bafs():
                author_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            else:
                author_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            author_paper_edges = torch.from_numpy(np.load(author_paper_edges_path))

            if is_this_machine_bafs():
                affiliation_author_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            else:
                affiliation_author_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            affiliation_author_edges = torch.from_numpy(np.load(affiliation_author_edges_path))

            if is_this_machine_bafs():
                paper_fos_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            else:
                paper_fos_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            paper_fos_edges = torch.from_numpy(np.load(paper_fos_edges_path))

        else:
            if is_this_machine_bafs():
                paper_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            else:
                paper_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            paper_paper_edges = torch.from_numpy(np.load(paper_paper_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                author_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            else:
                author_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            author_paper_edges = torch.from_numpy(np.load(author_paper_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                affiliation_author_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            else:
                affiliation_author_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            affiliation_author_edges = torch.from_numpy(np.load(affiliation_author_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                paper_fos_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            else:
                paper_fos_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            paper_fos_edges = torch.from_numpy(np.load(paper_fos_edges_path, mmap_mode='r'))


        if self.args.all_in_edges:
            graph_data = {
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('author', 'written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
                ('institute', 'affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
                ('fos', 'topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0])
            }
        else:
            graph_data = {
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
                ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
                ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
            }
        self.graph = dgl.heterograph(graph_data)     
        self.graph.predict = 'paper'

        if self.args.in_memory:
            if is_this_machine_bafs():
                paper_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper', 'node_feat.npy')
            else:
                paper_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper', 'node_feat.npy')
            paper_node_features = torch.from_numpy(np.load(paper_node_features_path))
            if self.args.num_classes == 19:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path(self.args.dataset_size, 'processed',
                    'paper', 'node_label_19.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                    'paper', 'node_label_19.npy')
                paper_node_labels = torch.from_numpy(np.load(paper_node_labels_path)).to(torch.long)  
            else:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path(self.args.dataset_size, 'processed',
                    'paper', 'node_label_2K.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                    'paper', 'node_label_2K.npy')
                paper_node_labels = torch.from_numpy(np.load(paper_node_labels_path)).to(torch.long)
        else:
            if is_this_machine_bafs():
                paper_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper', 'node_feat.npy')
            else:
                paper_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper', 'node_feat.npy')
            paper_node_features = torch.from_numpy(np.load(paper_node_features_path, mmap_mode='r'))
            if self.args.num_classes == 19:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path(self.args.dataset_size, 'processed',
                    'paper', 'node_label_19.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                    'paper', 'node_label_19.npy')
                paper_node_labels = torch.from_numpy(np.load(paper_node_labels_path, mmap_mode='r')).to(torch.long)  
            else:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path(self.args.dataset_size, 'processed',
                    'paper', 'node_label_2K.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                    'paper', 'node_label_2K.npy')
                paper_node_labels = torch.from_numpy(np.load(paper_node_labels_path, mmap_mode='r')).to(torch.long)                

        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        if self.args.in_memory:
            if is_this_machine_bafs():
                author_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author', 'node_feat.npy')
            else:
                author_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author', 'node_feat.npy')
            author_node_features = torch.from_numpy(np.load(author_node_features_path))
        else:
            if is_this_machine_bafs():
                author_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author', 'node_feat.npy')
            else:
                author_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author', 'node_feat.npy')
            author_node_features = torch.from_numpy(np.load(author_node_features_path, mmap_mode='r'))
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        if self.args.in_memory:
            if is_this_machine_bafs():
                institute_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            else:
                institute_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            institute_node_features = torch.from_numpy(np.load(institute_node_features_path))       
        else:
            if is_this_machine_bafs():
                institute_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            else:
                institute_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            institute_node_features = torch.from_numpy(np.load(institute_node_features_path, mmap_mode='r'))
        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        if self.args.in_memory:
            if is_this_machine_bafs():
                fos_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            else:
                fos_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            fos_node_features = torch.from_numpy(np.load(fos_node_features_path))       
        else:
            if is_this_machine_bafs():
                fos_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            else:
                fos_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            fos_node_features = torch.from_numpy(np.load(fos_node_features_path, mmap_mode='r'))
        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



class IGBHeteroDGLDatasetMassive(DGLDataset):
    """Compared with IGBHeteroDGLDataset, this class is essentially the same but introduced two features
    1) Use mmap to load features. This is optimization for massive dataset.
    2) Add the homogeneous graph construction option.
    """
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    def process(self):
        if self.args.in_memory:
            if is_this_machine_bafs():
                paper_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            else:
                paper_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            paper_paper_edges = torch.from_numpy(np.load(paper_paper_edges_path))

            if is_this_machine_bafs():
                author_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            else:
                author_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            author_paper_edges = torch.from_numpy(np.load(author_paper_edges_path))

            if is_this_machine_bafs():
                affiliation_author_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            else:
                affiliation_author_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            affiliation_author_edges = torch.from_numpy(np.load(affiliation_author_edges_path))

            if is_this_machine_bafs():
                paper_fos_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            else:
                paper_fos_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            paper_fos_edges = torch.from_numpy(np.load(paper_fos_edges_path))
        else:
            if is_this_machine_bafs():
                paper_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            else:
                paper_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__cites__paper', 'edge_index.npy')
            paper_paper_edges = torch.from_numpy(np.load(paper_paper_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                author_paper_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            else:
                author_paper_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__written_by__author', 'edge_index.npy')
            author_paper_edges = torch.from_numpy(np.load(author_paper_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                affiliation_author_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            else:
                affiliation_author_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'author__affiliated_to__institute', 'edge_index.npy')
            affiliation_author_edges = torch.from_numpy(np.load(affiliation_author_edges_path, mmap_mode='r'))

            if is_this_machine_bafs():
                paper_fos_edges_path = get_bafs_path(self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            else:
                paper_fos_edges_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__topic__fos', 'edge_index.npy')
            paper_fos_edges = torch.from_numpy(np.load(paper_fos_edges_path, mmap_mode='r'))

        if self.args.dataset_size == "full":
            num_paper_nodes = 269346174
            if self.args.dummy_feats:
                # Create features with hidden dimension as 1 to reduce memory cost during partitioning and load partition
                paper_node_features = torch.ones(num_paper_nodes, 1)
            else:
                if is_this_machine_bafs():
                    paper_node_features_path = get_bafs_path("full", 'processed',
                    'paper', 'node_feat.npy')
                else:
                    paper_node_features_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_feat.npy')
                paper_node_features = torch.from_numpy(np.memmap(paper_node_features_path, dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path("full", 'processed',
                    'paper', 'node_label_19.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_label_19.npy')
                paper_node_labels = torch.from_numpy(np.memmap(paper_node_labels_path, dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path("full", 'processed',
                    'paper', 'node_label_2K.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_label_2K.npy')
                paper_node_labels = torch.from_numpy(np.memmap(paper_node_labels_path, dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 277220883
            if self.args.dummy_feats:
                # Create features with hidden dimension as 1 to reduce memory cost during partitioning and load partition
                author_node_features = torch.ones(num_author_nodes, 1)
            else:
                if is_this_machine_bafs():
                    author_node_features_path = get_bafs_path("full", 'processed',
                    'author', 'node_feat.npy')
                else:
                    author_node_features_path = osp.join(self.dir, "full", 'processed',
                    'author', 'node_feat.npy')
                author_node_features = torch.from_numpy(np.memmap(author_node_features_path, dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        elif self.args.dataset_size == "large":
            num_paper_nodes = 100000000
            if self.args.dummy_feats:
                raise NotImplementedError("dummy_feats not implemented for large igbh")
            else:
                # The original code uses 'full' for the path. https://github.com/IllinoisGraphBenchmark/IGB-Datasets/blob/main/igb/dataloader.py#L321
                if is_this_machine_bafs():
                    paper_node_features_path = get_bafs_path("full", 'processed',
                    'paper', 'node_feat.npy')
                else:
                    paper_node_features_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_feat.npy')
                paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path("full", 'processed',
                    'paper', 'node_label_19.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_label_19.npy')
                paper_node_labels = torch.from_numpy(np.memmap(paper_node_labels_path, dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                if is_this_machine_bafs():
                    paper_node_labels_path = get_bafs_path("full", 'processed',
                    'paper', 'node_label_2K.npy')
                else:
                    paper_node_labels_path = osp.join(self.dir, "full", 'processed',
                    'paper', 'node_label_2K.npy')
                paper_node_labels = torch.from_numpy(np.memmap(paper_node_labels_path, dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 116959896
            if self.args.dummy_feats:
                raise NotImplementedError("dummy_feats not implemented for large igbh")
            else:
                if is_this_machine_bafs():
                    author_node_features_path = get_bafs_path("full", 'processed',
                    'author', 'node_feat.npy')
                else:
                    author_node_features_path = osp.join(self.dir, "full", 'processed',
                    'author', 'node_feat.npy')
                author_node_features = torch.from_numpy(np.memmap(author_node_features_path, dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        if self.args.dummy_feats:
            # Create features with hidden dimension as 1 to reduce memory cost during partitioning and load partition
            if self.args.dataset_size == "full":
                num_fos_nodes = 712960
                num_institute_nodes = 26918
            else:
                raise NotImplementedError
            institute_node_features = torch.ones(num_institute_nodes, 1)
            fos_node_features = torch.ones(num_fos_nodes, 1)
        else:
            if is_this_machine_bafs():
                institute_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            else:
                institute_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'institute', 'node_feat.npy')
            institute_node_features = torch.from_numpy(np.load(institute_node_features_path, mmap_mode='r'))

            if is_this_machine_bafs():
                fos_node_features_path = get_bafs_path(self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            else:
                fos_node_features_path = osp.join(self.dir, self.args.dataset_size, 'processed',
                'fos', 'node_feat.npy')
            fos_node_features = torch.from_numpy(np.load(fos_node_features_path, mmap_mode='r'))
        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': len(institute_node_features), 'fos': len(fos_node_features)}


        if self.args.load_homo_graph:
            from .utils import construct_homogeneous
            # TODO: This part is faulty! It did not contain necessary nodes data and edges data, involving feat, label, etc.
            # Sorted the key according to dictionary order, i.e., the order in graph.canonical_etypes
            graph_data = {
                ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
                ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            }
            self.graph = construct_homogeneous(graph_data,[('author', 'affiliated_to', 'institute'), ('paper', 'cites', 'paper'),('paper', 'topic', 'fos'),('paper', 'written_by', 'author')],['author', 'fos', 'institute', 'paper'],{
                'author':num_author_nodes,
                'fos':num_fos_nodes,
                'institute':num_institute_nodes,
                'paper':num_paper_nodes
            })
        
        
        else:
            # Sorted the key according to dictionary order, i.e., the order in graph.canonical_etypes
            graph_data = {
                ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
                ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
                ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
                ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            }
            self.graph = dgl.heterograph(graph_data, num_nodes_dict)  
            # self.graph.canonical_etypes == [('author', 'affiliated_to', 'institute'), ('paper', 'cites', 'paper'), ('paper', 'topic', 'fos'), ('paper', 'written_by', 'author')]
            # self.graph.ntypes == ['author', 'fos', 'institute', 'paper']
            self.graph.predict = 'paper'
            self.graph = dgl.remove_self_loop(self.graph, etype='cites')
            self.graph = dgl.add_self_loop(self.graph, etype='cites')
            self.graph.nodes['paper'].data['feat'] = paper_node_features
            self.graph.num_paper_nodes = paper_node_features.shape[0]
            self.graph.nodes['paper'].data['label'] = paper_node_labels
            self.graph.nodes['author'].data['feat'] = author_node_features
            self.graph.num_author_nodes = author_node_features.shape[0]

            self.graph.nodes['institute'].data['feat'] = institute_node_features
            self.graph.num_institute_nodes = institute_node_features.shape[0]

            self.graph.nodes['fos'].data['feat'] = fos_node_features
            self.graph.num_fos_nodes = fos_node_features.shape[0]
            
            n_nodes = paper_node_features.shape[0]

            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.nodes['paper'].data['train_mask'] = train_mask
            self.graph.nodes['paper'].data['val_mask'] = val_mask
            self.graph.nodes['paper'].data['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/gnndataset', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=1,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--all_in_edges', type=bool, default=True, 
        help="Set to false to use default relation. Set this option to True to use all the relation types in the dataset since DGL samplers require directed in edges.")
    args = parser.parse_args()

    dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]
    print(g)
    homo_g = dgl.to_homogeneous(g)
    print()
    print(homo_g)
    print()
