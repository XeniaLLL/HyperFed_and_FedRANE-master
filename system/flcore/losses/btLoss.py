from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class BTLoss(nn.Module):
    def __init__(self, lambd=1.):
        '''

        Args:
            lambd: weight for off-diagonal elements
        '''
        super(BTLoss, self).__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        bs = z1.shape[0]
        dim = z1.shape[-1]
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        c = torch.mm(z1_norm.T, z2_norm) / bs
        c_diff = (c - torch.eye(dim)).pow(2)
        c_diff_off_digonal = c_diff.flatten()[1:].view(dim - 1, dim + 1)[:, :-1].reshape(dim, dim - 1)
        lambd_c_diff = c_diff_off_digonal.mul_(self.lambd)
        return lambd_c_diff.sum()


class MECLoss(nn.Module):
    def __init__(self, mu, lambd, n):
        '''

        Args:
            mu: constant related to m and d, (m+d)/2
            lambd: hyper-param determined by distortion, d/m/(\epsilon)^2
            n: order for Tylar expansion
        '''
        super(MECLoss, self).__init__()
        self.mu = mu
        self.lambd = lambd
        self.n = n

    def forward(self, z1, z2):
        c = self.lambd * (torch.mm(z1, z2.T))  # batch-wise
        # c= self.lambd * (torch.mm(z1.T,z2)) # feature-wise
        power = c
        sum_p = torch.zeros_like(power)
        for k in range(1, self.n + 1):
            if k > 1:
                power = torch.mm(power, c)
            if (k + 1) % 2 == 0:
                sum_p += power / k
            else:
                sum_p -= power / k
        loss = -self.mu * torch.trace(sum_p)
        return loss


class AULoss(nn.Module):
    def __init__(self, alpha=2, t=2):
        super(AULoss, self).__init__()
        self.alpha = alpha
        self.t = t

    def align_loss(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(self.alpha).mean()

    def uniform_loss(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-self.t).exp().mean().log()

    def forward(self, z1, z2):
        loss_align = self.align_loss(z1, z2)
        uniform_z1, uniform_z2 = self.uniform_loss(z1), self.uniform_loss(z2)

        return loss_align.mean() + ((uniform_z1 + uniform_z2) / 2)


def mixup_data(x, y, alpha=1.):
    if alpha > 0:
        lambd = np.random.beta(alpha, alpha)
    else:
        lambd = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lambd * x + (1 - lambd) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lambd


class MixupLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(MixupLoss, self).__init__()
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss()

    def forward(self, mixed_preds, y1, y2, lambd):
        # mixed_x, y_a, y_b, lambd = mixup_data(x,y, self.alpha)
        return lambd * self.loss(mixed_preds, y1) + (1 - lambd) * self.loss(mixed_preds, y2)

