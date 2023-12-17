import numpy as np
import torch
import torch.nn as nn
import geoopt
import torch.backends.cudnn
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from PIL import Image
from PIL import ImageEnhance
import torchvision
from torchvision import transforms


def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
    """
    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Project a point from Klein model to Poincare model
def k2p(x, c):
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Project a point from Poincare model to Klein model
def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def poincare_mean(x, dim=0, c=1.0):
    # To calculate the mean, another model of hyperbolic space named Klein model is used.
    # 1. point is projected from Poincare model to Klein model using p2k, output x is a point in Klein model
    x = p2k(x, c)
    # 2. mean is calculated
    lamb = lorenz_factor(x, c=c, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(lamb, dim=dim, keepdim=True)
    # 3. Mean is projected from Klein model to Poincare model
    mean = k2p(mean, c)
    return mean.squeeze(dim)


# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x


class ProtoNet(nn.Module):
    def __init__(self, shot=1, way=5, curvature=1, temperature=0.5, dim=1600, is_hyperbolic=True):
        super().__init__()

        self.shot = shot
        self.way = way
        self.c = curvature
        self.temperature = temperature
        self.is_hyperbolic = is_hyperbolic

        # Base Model: ConvNet
        self.encoder = ConvNet(z_dim=dim)

        # If working in Hyperbolic Space
        if self.is_hyperbolic:
            self.manifold = geoopt.PoincareBall(c=self.c)

    def forward(self, data_shot, data_query):
        # 1. feed data to the model
        proto = self.encoder(data_shot)

        # Hyperbolic Space:
        if self.is_hyperbolic:
            # 2. encoder is Euclidean, so proto is in Euclidean space and should be projected to Hyperboolic space using exponential map
            proto = self.manifold.expmap0(proto)

            proto = proto.reshape(self.shot, self.way, -1)

            # 3. calculate prototypes based on mean of data
            proto = poincare_mean(proto, dim=0, c=self.manifold.c.item())

            # 4. query is projected to hyperbolic space too
            data_query = self.manifold.expmap0(self.encoder(data_query))

            # 5. Logits is calculated based on the Hyperbolic distance between data query and proto
            logits = (-self.manifold.dist(data_query[:, None, :], proto) / self.temperature)

        # Euclidean Space
        else:
            # 2. calculate prototypes based on mean of data
            proto = proto.reshape(self.shot, self.way, -1).mean(dim=0)

            # 3. Logits is calculated based on the Euclidean distance between data query and proto
            logits = (((self.encoder(data_query)[:, None, :] - proto) ** 2).sum(dim=-1) / self.temperature)
        return logits
