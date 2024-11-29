import argparse

import numpy as np
import torch

from LEGNN import PLGCN
from dataset.dataset import load_data_set, noisify_labels
from utils.utils import init_gpuseed


def main(args):
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    # data load
    adj, features, labels, idx_train, idx_val, idx_test, nclass, data = load_data_set(root='./data', name=args.dataset,
                                                                                      train_rate=args.label_rate,
                                                                                      seed=0)
    noise_labels, noise_idx, clean_idx = noisify_labels(labels, idx_train, idx_val, nclass, noise_type=args.noise,
                                                        ptb=args.ptb_rate, seed=0)

    model = PLGCN(nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  self_loop=True,
                  dropout=args.dropout,
                  device=device).to(device)
    model.fit(features, adj, noise_labels, idx_train, idx_val, mask_rate=args.mask_rate, mask_times=args.mask_times,
              pre_lr=args.pre_lr,
              pre_weight_decay=args.pre_weight_decay,
              nll_lr=args.nll_lr,
              nll_weight_decay=args.nll_weight_decay)
    return model.test(idx_test)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4, help='Random seed.')
    parser.add_argument('--gpu', help='used gpu id', default='6', type=str, required=False)
    parser.add_argument('--dataset', type=str, default="citeseer",
                        choices=['cora', 'citeseer', 'pubmed', 'dblp', 'ogbn-arxiv', 'polblogs', 'cora_ml'],
                        help='dataset')
    parser.add_argument("--label_rate", type=float, default=0.05,
                        help='rate of labeled data')
    parser.add_argument('--ptb_rate', type=float, default=0.4,
                        help="noise ptb_rate")
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')
    parser.add_argument('--mask_rate', type=float, default=0.5,
                        help='threshold of mask rate')
    parser.add_argument('--mask_times', type=int, default=10,
                        help='threshold of mask times')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--pre_lr', type=float, default=2,
                        help='Initial learning rate.')
    parser.add_argument('--pre_weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nll_lr', type=float, default=2,
                        help='Initial learning rate.')
    parser.add_argument('--nll_weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    args = parser.parse_known_args()[0]

    if args.dataset == 'citeseer':
        if args.noise == 'pair':
            args.pre_lr = 1
            args.pre_weight_decay = 5e-3
            args.nll_lr = 2
            args.nll_weight_decay = 1e-2

    if args.dataset == 'cora':
        if args.noise == 'uniform':
            args.pre_lr = 1
            args.pre_weight_decay = 1e-3
            args.nll_lr = 2
            args.nll_weight_decay = 5e-3
        if args.noise == 'pair':
            args.mask_rate = 0.8
            args.mask_times = 20
            args.pre_lr = 0.05
            args.pre_weight_decay = 1e-2
            args.nll_lr = 2
            args.nll_weight_decay = 5e-3

    if args.dataset == 'ogbn-arxiv':
        if args.noise == 'uniform':
            args.mask_rate = 0.8
            args.pre_lr = 0.5
            args.pre_weight_decay = 5e-4
            args.nll_lr = 0.5
            args.nll_weight_decay = 5e-4
        if args.noise == 'pair':
            args.mask_rate = 0.8
            args.pre_lr = 0.5
            args.pre_weight_decay = 5e-4
            args.nll_lr = 0.5
            args.nll_weight_decay = 1e-3
    if args.dataset == 'pubmed':
        if args.noise == 'uniform':
            args.pre_lr = 0.5
            args.pre_weight_decay = 1e-3
            args.nll_lr = 0.5
            args.nll_weight_decay = 1e-3
        if args.noise == 'pair':
            args.pre_lr = 0.5
            args.pre_weight_decay = 5e-4
            args.nll_lr = 0.5
            args.nll_weight_decay = 5e-4
    main(args=args).cpu()
