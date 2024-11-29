import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, one_hot


class PLGCN(nn.Module):
    def __init__(
            self,
            nfeat,
            nhid,
            nclass,
            dropout=0.5,
            with_relu=True,
            with_bias=True,
            self_loop=True,
            device=None,
    ):
        super(PLGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.labels_com = None
        self.labels_multi = None
        self.labels = None
        self.innode2index = None
        self.best_acc_val = None

    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x = F.relu(self.gc1(x, edge_index, edge_weight))
        else:
            x = self.gc1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return x

    def initialize(self):
        """Initialize parameters of GCN."""
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val, pre_lr, pre_weight_decay, nll_lr, nll_weight_decay,
            mask_rate=0.8, mask_times=10, train_iters=200, initialize=True):

        if initialize:
            self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)

        # The index of the incoming arc.
        innode2index = {}
        for i in range(len(self.edge_index[1])):
            col = self.edge_index[1][i].item()
            innode2index[col] = innode2index.get(col, [])
            innode2index[col].append(i)
        self.innode2index = innode2index

        self.edge_index, self.edge_weight = self.edge_index.to(
            self.device
        ), self.edge_weight.float().to(self.device)
        if sp.issparse(features):
            features = (
                utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
            )
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)
        self.labels_multi, self.labels_com = self.pre_labeling(self.labels, idx_train, idx_val, train_iters,
                                                               pre_lr, pre_weight_decay,
                                                               mask_rate, mask_times)
        self.initialize()
        self.train_lnl(self.labels, idx_train, idx_val, train_iters, nll_lr, nll_weight_decay, mask_rate, mask_times)

    def pre_labeling(self, labels, idx_train, idx_val, train_iters, lr, weight_decay, mask_rate, mask_times):
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        best_acc_val = 0
        best_iter = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_iter = i
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        self.best_acc_val = best_acc_val

        print(f"Best iter is {best_iter}")

        # multi-labeling
        self.eval()
        labels_multi, labels_com = self.multi_labeling(mask_ratio=mask_rate, mask_times=mask_times, idx_train=idx_train)
        print(labels_multi.sum(1).mean())
        train_labels_oh = one_hot(index=labels, num_classes=self.nclass, dtype=torch.int32)
        train_labels_oh = train_labels_oh.bool()
        print(labels_multi[train_labels_oh].float().mean())
        return labels_multi, labels_com

    def multi_labeling(self, mask_ratio, mask_times, idx_train, edge_mask_only=True):
        labels = self.labels
        self.eval()
        labels_multi = torch.zeros([self.features.shape[0], self.nclass]).to(self.device)
        labels_com = torch.zeros([self.features.shape[0], self.nclass]).to(self.device)

        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.pre_probs = F.softmax(output, dim=1)
        preds = output.max(1)[1]
        preds_oh = one_hot(index=preds, num_classes=self.nclass, dtype=torch.int32)
        labels_multi += preds_oh

        com_preds = output.min(1)[1]
        com_oh = one_hot(index=com_preds, num_classes=self.nclass, dtype=torch.int32)
        labels_com += com_oh

        for _ in range(mask_times):
            masked_indices = []
            edge_weight_term = deepcopy(self.edge_weight)
            for innode_index in self.innode2index.values():
                masked_indices.extend(
                    random.sample(innode_index, int(mask_ratio * len(innode_index)))
                )
            edge_weight_term[masked_indices] = 0
            feartures_term = deepcopy(self.features)
            if not edge_mask_only:
                feartures_term = self.feature_mask(0.3)

            output = self.forward(feartures_term, self.edge_index, edge_weight_term)

            preds = output.max(1)[1]
            preds_oh = one_hot(index=preds, num_classes=self.nclass, dtype=torch.int32)
            labels_multi += preds_oh

            com_preds = output.min(1)[1]
            com_oh = one_hot(index=com_preds, num_classes=self.nclass, dtype=torch.int32)
            labels_com += com_oh

        train_labels_oh = one_hot(index=labels[idx_train], num_classes=self.nclass, dtype=torch.int32)
        labels_multi[idx_train] += train_labels_oh
        labels_multi[labels_multi > 1] = 1
        assert labels_multi.sum(1).min() >= 1.0
        return labels_multi, labels_com

    def feature_mask(self, mask_ratio):  ### 按比例噪声注入？
        feartures_term = deepcopy(self.features)
        masked_matrix = feartures_term.mean(0)
        masked_matrix = masked_matrix.repeat(len(feartures_term), 1)
        masked_inds = torch.rand_like(feartures_term)
        masked_inds[masked_inds < mask_ratio] = -1
        masked_inds[masked_inds >= mask_ratio] = 0
        masked_inds[masked_inds == -1] = 1
        print("Real Feature Masked Rate: ", masked_inds.mean())
        return feartures_term * (1 - masked_inds) + masked_matrix * masked_inds

    def probs_weighted_ce(self, probs, weight):
        out_log = torch.log(probs + 1e-8)
        loss = -(out_log * weight).sum(dim=1)
        return loss.mean()

    def update_weight(self, Y, probs):
        weight = Y * probs
        sum_weight = weight.sum(dim=1).unsqueeze(1).repeat(1, weight.shape[1])
        weight = torch.div(weight, sum_weight).detach_()
        return weight

    def symmetric_loss(self, outputs, partial_weight, com_weight):
        probs = F.softmax(outputs, dim=1)
        partial_loss = self.probs_weighted_ce(probs, partial_weight)
        com_loss = self.probs_weighted_ce(1 - probs, com_weight)
        return partial_loss + com_loss

    def train_lnl(self, labels, idx_train, idx_val, train_iters, lr, weight_decay, mask_ratio, mask_times):
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

        best_loss_val = 100
        best_acc_val = 0
        best_iter = 0

        partial_weight = deepcopy(self.labels_multi)
        com_weight = deepcopy(self.labels_com)
        partial_weight = self.update_weight(partial_weight, self.pre_probs)
        com_weight = self.update_weight(com_weight, 1 - self.pre_probs)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = self.symmetric_loss(output, partial_weight, com_weight)
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            loss_val = self.symmetric_loss(output[idx_val], self.labels_multi[idx_val], self.labels_com[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                probs = F.softmax(output, dim=1)
                if i >= 5:
                    partial_weight = self.update_weight(self.labels_multi, probs)
                    com_weight = self.update_weight(self.labels_com, 1 - probs)
                best_iter = i
                self.output = output
                weights = deepcopy(self.state_dict())
                if acc_val >= self.best_acc_val:
                    self.labels_multi, self.labels_com = self.multi_labeling(mask_ratio=mask_ratio,
                                                                             mask_times=mask_times,
                                                                             idx_train=idx_train)
        print(f"Best iter is {best_iter}")
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        loss_test = F.cross_entropy(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )
        return acc_test
