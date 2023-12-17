#
# Obtain hyperspherical prototypes prior to network training.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#
import math
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


#
# PArse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
    parser.add_argument('-c', dest="classes", default=10, type=int)
    parser.add_argument('-d', dest="dims", default=20, type=int) # careful
    parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=5000, type=int, )
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="../../prototypes", type=str)
    parser.add_argument('-w', dest="wtvfile", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    parser.add_argument('-mani', dest="manifold", default="hyperbolic", type=str) # hyperspherical hyperbolic
    args = parser.parse_args()
    return args


#
# Compute the loss related to the hyper-spherical prototypes
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cos sim
    product = torch.matmul(prototypes, prototypes.T) + 1
    # remove diagonal from loss
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum consine similarity
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()  # hypersphere get the max val


#
# unify
#
def prototype_unify(num_classes):
    single_angle = 2 * math.pi / num_classes
    help_list = np.array(range(0, num_classes))
    angles = (help_list * single_angle).reshape(-1, 1)
    sin_points = np.sin(angles)
    cos_points = np.cos(angles)
    set_prototypes = torch.tensor(np.concatenate((cos_points, sin_points), axis=1))
    return set_prototypes


#
# Compute the semantic relation loss
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.T) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:, 0], triplets[:, 1]]
    loss2 = product[triplets[:, 2], triplets[:, 3]]
    return loss1.mean() + loss2.mean(), product.max()


#
# Generating prototypes
#
if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 64, 'pin_memory': True}

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # init prototypes and optimizer
    if os.path.exists(args.wtvfile):  # note use pretrained word embeddings as semantic information
        user_wtv = True
        wtvv = np.load(args.wtvfile)
        for i in range(wtvv.shape[0]):
            wtvv[i] /= np.linalg.norm(wtvv[i])
        wtvv = torch.from_numpy(wtvv.vectors)  # careful unk data for converting
        wtvsim = torch.matmul(wtvv, wtvv.T).float()

        # precompute triplets
        nns, others = [], []
        for i in range(wtvv.shape[0]):
            sorder = np.argsort(wtvsim[i, :])[::-1]
            nns.append(sorder[:args.nn])
            others.append(sorder[args.nn:-1])

        triplets = []
        for i in range(wtvv.shape[0]):
            for j in range(len(nns[i])):
                for k in range(len(others[i])):
                    triplets.append([i, j, i, k])
        triplets = np.array(triplets).astype(int)
    else:
        use_wtv = False
        triplets = None

    # Init prototypes
    prototypes = torch.randn(args.classes, args.dims)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)

    # optimize for separation
    if args.dims > 2:
        for i in range(args.epochs):
            # compute loss
            loss1, sep = prototype_loss(prototypes)  # note 直接优化,没有语义信息作为privileged info
            print(loss1, sep)

            if use_wtv:
                loss2 = prototype_loss_sem(prototypes, triplets)
                loss = loss1 + loss2
            else:
                loss = loss1

            # update
            loss.backward()
            optimizer.step()
            # renormalze prototypes
            prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
            optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)
            print(f"{i}/{args.epochs}: {sep}")
    elif args.dims == 2:
        prototypes = prototype_unify(args.classes)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    else:
        raise Exception("Dimension is incorrect")

    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)
    # save
    np.save(os.path.join(args.resdir, f"{args.manifold}-prototypes-{args.dims}-{args.classes}.npy"), prototypes.data.numpy())
