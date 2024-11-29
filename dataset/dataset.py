import os.path as osp
import pickle as pkl
import sys
import urllib.request

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import ClusterData
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

from utils.utils import get_train_val_test, noisify_train_val


class Dataset:
    """Dataset class contains four citation network datasets "cora", "cora-ml", "citeseer" and "pubmed",
    and one blog dataset "Polblogs".
    The 'cora', 'cora-ml', 'poblogs' and 'citeseer' are downloaded from https://github.com/danielzuegner/gnn-meta-attack/tree/master/data, and 'pubmed' is from https://github.com/tkipf/gcn/tree/master/gcn/data.

    Parameters
    ----------
    root :
        root directory where the dataset should be saved.
    name :
        dataset name, it can be choosen from ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed']
    seed :
        random seed for splitting training/validation/test.
    --------
	We can first create an instance of the Dataset class and then take out its attributes.

	>>> from deeprobust.graph.data import Dataset
	>>> data = Dataset(root='/tmp/', name='cora')
	>>> adj, features, labels = data.adj, data.features, data.labels
	>>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    """

    def __init__(self, root, name, train_rate, seed=None):
        import os

        path = root

        if not os.path.exists(path):
            os.makedirs(path)
        self.name = name.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed'], \
            'Currently only support cora, citeseer, cora_ml, polblogs, pubmed'

        self.seed = seed

        self.url = 'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'

        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test(train_rate)

    def get_train_val_test(self, train_rate):
        val_size = 0.2 - train_rate
        return get_train_val_test(nnodes=self.adj.shape[0], val_size=val_size, test_size=0.8, stratify=self.labels,
                                  seed=self.seed)

    def load_data(self):
        if self.name == 'pubmed':
            return self.load_pubmed()

        if not osp.exists(self.data_filename):
            self.download_npz()

        self.adj, self.features, self.labels = self.load_npz(self.data_filename)
        adj, features, labels, _ = get_adj(self.adj, self.features, self.labels)
        return adj, features, labels

    def download_npz(self):
        """Download adjacen matrix npz file from self.url.
        """
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception(
                '''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        try:
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
        except:
            raise Exception(
                '''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_file = "ind.{}.test.index".format(dataset)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                     loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                              loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx


def ogbn_dataset(root, name):
    dataset = PygNodePropPredDataset(root=root, name=name)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]  # pyg graph object
    x = graph.x
    y = graph.y
    adj = graph.edge_index
    adj = to_scipy_sparse_matrix(adj)
    features = x.numpy()
    labels = y.numpy().squeeze(-1)
    adj, features, labels, lcc = get_adj(adj, features, labels)
    all_selected = np.zeros(len(features), dtype=bool)
    all_selected[lcc] = True
    train_idx = train_idx[all_selected[train_idx]]
    valid_idx = valid_idx[all_selected[valid_idx]]
    test_idx = test_idx[all_selected[test_idx]]
    graph.x = torch.from_numpy(features)
    graph.y = torch.from_numpy(labels.reshape(-1, 1))
    graph.edge_index = from_scipy_sparse_matrix(adj)
    return adj, features, labels, train_idx.numpy(), valid_idx.numpy(), test_idx.numpy(), graph


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def largest_connected_components(adj, n_components=1):
    """Select k largest connected components.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        input adjacency matrix
    n_components : int
        n largest connected components we want to select
    """

    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def get_adj(adj, features, labels):
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    lcc = largest_connected_components(adj)
    adj_csr = adj.tocsr()
    adj_csr = adj_csr[lcc][:, lcc]
    adj = adj_csr.tocoo()
    features = features[lcc]
    labels = labels[lcc]
    assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels, lcc


def load_data_set(root='./data', name=None, train_rate=0.05, seed=0):
    print('Loading {} dataset...'.format(name))
    if name in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed']:
        data = Dataset(root=root, name=name, train_rate=train_rate, seed=seed)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif name in ['ogbn-arxiv']:
        adj, features, labels, idx_train, idx_val, idx_test, data = ogbn_dataset(root, name)
    else:
        raise ValueError('Error dataset')
    nclass = labels.max() + 1
    return adj, features, labels, idx_train, idx_val, idx_test, nclass, data


def load_clustered_ogbn(data, idx_train, idx_val, idx_test, labels, root='./data', num_parts=2):
    labels = data.labels if labels is None else labels
    data.y = torch.tensor(labels.reshape(-1, 1))
    idx_all = torch.zeros(len(labels), dtype=torch.uint8) + 4
    idx_all[idx_train] = 0
    idx_all[idx_val] = 1
    idx_all[idx_test] = 2
    assert idx_all.max() < 4
    data.idx_split = idx_all
    cluster_data = ClusterData(data, num_parts=num_parts, save_dir=root, recursive=True)
    return cluster_data


def noisify_labels(labels, idx_train, idx_val, nclass, noise_type, ptb=0.4, seed=0):
    train_labels = labels[idx_train]
    val_labels = labels[idx_val]
    train_val_labels = np.concatenate([train_labels, val_labels], axis=0)
    idx = np.concatenate([idx_train, idx_val], axis=0)
    noise_y, P, noise_idx, clean_idx = noisify_train_val(train_val_labels, idx_train.shape[0], nclass, ptb, seed,
                                                         noise_type)
    noise_labels = labels.copy()
    noise_labels[idx] = noise_y
    return noise_labels, noise_idx, clean_idx
