import torch
from torch import nn


class CosTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosTripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, x_pos, x_neg):
        ap_distance = self.loss(anchor, x_pos)
        an_distance = self.loss(anchor, x_neg)
        loss = torch.maximum(an_distance - ap_distance + self.margin, torch.tensor(0.))
        return loss


class ArcCoshLoss(nn.Module):
    def __init__(self, manifold):
        super(ArcCoshLoss, self).__init__()
        self.manifold = manifold

    def forward(self, x, y):
        # x_y = (x - y).pow(2).sum(dim=-1)
        # x_norm = (1 - (x).pow(2).sum(dim=-1))
        # y_norm = (1 - (y).pow(2).sum(dim=-1))
        # dis = 1 + 2 * torch.div(x_y, torch.mul(x_norm, y_norm) + 1e-8)  # urgent todo norm 部分
        # return self.loss(dis)
        dist = self.manifold.dist(x, y, dim=-1).mean()
        return dist


class InfoNCE(nn.Module):
    def __init__(self, manifold, tau=0.1):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.manifold = manifold
        self.CE = nn.CrossEntropyLoss()

    def forward(self, x, y, y_neg):
        batch_n_neg_classes, _ = y_neg.shape
        batch = y.shape[0]
        loss_pos = self.manifold.dist(x, y).reshape(-1, 1)
        x_neg = x.repeat(1, batch_n_neg_classes // batch).reshape(y_neg.shape)
        loss_neg = self.manifold.dist(x_neg, y_neg)
        loss_neg = loss_neg.reshape(batch, -1)
        logits = torch.cat((loss_pos, loss_neg), dim=1)
        logits /= (-1) * self.tau
        target = torch.zeros(batch, device=x.device).long()
        loss = self.CE(logits, target)  # todo careful
        return loss


class ACoshTripletLoss(nn.Module):  # todo
    def __init__(self, manifold, margin=1.):
        super(ACoshTripletLoss, self).__init__()
        self.margin = margin
        self.manifold = manifold

    def forward(self, anchor, x_pos, x_neg):
        ap_distance = self.manifold.dist(anchor, x_pos)
        an_distance = self.manifold.dist(anchor, x_neg)
        # loss = torch.maximum(an_distance - ap_distance + self.margin, torch.tensor(0.))
        loss = torch.maximum(ap_distance - an_distance + self.margin, torch.tensor(0.))
        return loss
