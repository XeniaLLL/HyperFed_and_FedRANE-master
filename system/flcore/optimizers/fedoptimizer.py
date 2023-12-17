import torch
from torch.optim import Optimizer
from torch.optim.adam import *
from torch.optim.sgd import *
from geoopt.optim import RiemannianAdam, RiemannianSGD

class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if (beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data.add_(- group['lr'] * (p.grad.data + group['eta'] * \
                                             self.server_grads[i] - self.pre_grads[i]))
                i += 1


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (
                        p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


# class pFedMeOptimizer(Optimizer):
#     def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
#         #self.local_weight_updated = local_weight # w_i,K
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         defaults = dict(lr=lr, lamda=lamda, mu = mu)
#         super(pFedMeOptimizer, self).__init__(params, defaults)

#     def step(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
#         return  group['params'], loss

#     def update_param(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = localweight.data
#         #return  p.data
#         return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)
        # default.update(**kwargs)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:  # ignore the computation for fixed layers (the differences are always 0.)
                    continue
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (
                        p.data - g.data)  # note: adaptive mixup parameter betwn global & local
                p.data.add_(d_p, alpha=-group['lr'])


class PerturbedSGDGradientDescent(SGD):
    def __init__(self, params, lr=0.001, mu=0, weight_decay=0,momentum=0.,  **kwargs):
        super(PerturbedSGDGradientDescent, self).__init__(params=params, lr=lr,
                                                           weight_decay=weight_decay,momentum=momentum, **kwargs)
        self.param_groups[0].update(dict(mu=mu))


    @torch.no_grad()
    def step(self, global_params, device, closure=None, *, grad_scaler=None):
        super(PerturbedSGDGradientDescent, self).step()
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:  # ignore the computation for fixed layers (the differences are always 0.)
                    continue
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (
                        p.data - g.data)  # note: adaptive mixup parameter betwn global & local
                p.data.add_(d_p, alpha=-group['lr'])


class PerturbedAdamGradientDescent(Adam):
    def __init__(self, params, lr=0.001, mu=0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, **kwargs):
        super(PerturbedAdamGradientDescent, self).__init__(params=params, lr=lr, betas=betas, eps=eps,
                                                           weight_decay=weight_decay, **kwargs)
        self.param_groups[0].update(dict(mu=mu))


    @torch.no_grad()
    def step(self, global_params, device, closure=None, *, grad_scaler=None):
        super(PerturbedAdamGradientDescent, self).step()
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:  # ignore the computation for fixed layers (the differences are always 0.)
                    continue
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (
                        p.data - g.data)  # note: adaptive mixup parameter betwn global & local
                p.data.add_(d_p, alpha=-group['lr'])



class PerturbedRiemannianSGDGradientDescent(RiemannianSGD):
    def __init__(self, params, lr=0.001, mu=0, weight_decay=0,momentum=0.,  **kwargs):
        super(PerturbedRiemannianSGDGradientDescent, self).__init__(params=params, lr=lr, momentum=momentum,
                                                           weight_decay=weight_decay, **kwargs)
        self.param_groups[0].update(dict(mu=mu))


    @torch.no_grad()
    def step(self, global_params, device, closure=None, *, grad_scaler=None):
        super(PerturbedRiemannianSGDGradientDescent, self).step()
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:  # ignore the computation for fixed layers (the differences are always 0.)
                    continue
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (
                        p.data - g.data)  # note: adaptive mixup parameter betwn global & local
                p.data.add_(d_p, alpha=-group['lr'])



class PerturbedRiemannianAdamGradientDescent(RiemannianAdam):
    def __init__(self, params, lr=0.001, mu=0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, **kwargs):
        super(PerturbedRiemannianAdamGradientDescent, self).__init__(params=params, lr=lr, betas=betas, eps=eps,
                                                           weight_decay=weight_decay, **kwargs)
        self.param_groups[0].update(dict(mu=mu))


    @torch.no_grad()
    def step(self, global_params, device, closure=None, *, grad_scaler=None):
        super(PerturbedRiemannianAdamGradientDescent, self).step()
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:  # ignore the computation for fixed layers (the differences are always 0.)
                    continue
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (
                        p.data - g.data)  # note: adaptive mixup parameter betwn global & local
                p.data.add_(d_p, alpha=-group['lr'])


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, global_control: torch.nn.Module, local_control: torch.nn.Module):
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], global_control.parameters(), local_control.parameters()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']



