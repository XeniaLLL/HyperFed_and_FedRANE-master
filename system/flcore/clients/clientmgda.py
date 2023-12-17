import torch
import torch.nn as nn
import numpy as np
import time

from flcore.clients.clientbase_cl import ClientCLY
from flcore.losses.costripletLoss import ArcCoshLoss, ACoshTripletLoss
from flcore.losses.btLoss import MixupLoss, mixup_data
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from geoopt import PoincareBall
import geoopt
from torch.autograd import Variable


def module_train_(module: nn.Module, mode: bool = True):
    module.train(mode)
    for param in module.parameters():
        param.requires_grad_(mode)


class ClientMGDA(ClientCLY):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super(ClientMGDA, self).__init__(args, id, train_samples, test_samples, **kwargs)
        self.ball = PoincareBall(args.curvature)
        self.loss6 = ACoshTripletLoss(manifold=self.ball, margin=args.margin_triplet)
        self.loss8 = MixupLoss(alpha=0.5)

        self.optimizer = geoopt.optim.RiemannianSGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                                    weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=750,
                                                         gamma=0.1)
        self.fine_tuning_steps = args.fine_tuning_steps
        self.polars = args.predictor.weight.data

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # avgloss, avglosscount, newloss, acc, newacc = 0., 0, 0., 0., 0.
        for step in range(max_local_steps):
            for i, (x_pos, y, y_neg) in enumerate(trainloader):
                x_pos = x_pos.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                y_exp_map = self.polars[y]
                output_pos = self.model.base(x_pos)
                output_pos_exp_map = self.ball.expmap0(output_pos)

                # loss 6
                y_neg_exp_map = self.polars[y_neg]
                loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()

                # loss8
                mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
                output_mixed = self.model.base(mixed_x)
                output_mixed_exp_map = self.ball.expmap0(output_mixed)
                normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
                preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
                loss8 = self.loss8(preds_mix, y_a, y_b, lambd)

                # loss = loss6
                loss = loss6 + 2 * loss8  # careful

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_gradient(self):
        # calculate the averaged gradient for each batch
        grads = {}
        trainloader = self.load_train_data()
        for i, (x_pos, y, y_neg) in enumerate(trainloader):
            x_pos = x_pos.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_exp_map = self.polars[y]
            output_pos = self.model.base(x_pos)
            output_pos_exp_map = self.ball.expmap0(output_pos)

            # loss 6
            y_neg_exp_map = self.polars[y_neg]
            loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()
            # loss8
            mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
            output_mixed = self.model.base(mixed_x)
            output_mixed_exp_map = self.ball.expmap0(output_mixed)
            normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
            preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
            loss8 = self.loss8(preds_mix, y_a, y_b, lambd)

            # loss = loss6
            loss = loss6 + 2 * loss8  # careful
            loss.backward()
            for idx, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    grads[idx] = grads.get(idx, 0) + Variable(param.grad.data.clone(), requires_grad=False)

        gradient = []
        for idx in grads:
            gradient.append(grads[idx] / len(trainloader))

        return gradient


    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        loss = 0
        with torch.no_grad():
            for i, (x_pos, y, y_neg) in enumerate(trainloader):
                x_pos = x_pos.to(self.device)
                train_num += y.shape[0]
                y = y.to(self.device)

                y_exp_map = self.polars[y]  # note refer to prototype
                output_pos = self.model.base(x_pos)  # note E representation
                output_pos_exp_map = self.ball.expmap0(output_pos)  # project on the hyperbolic


                # loss 6
                y_neg_exp_map = self.polars[y_neg]
                loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()

                # loss 8
                mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
                output_mixed = self.model.base(mixed_x)  # note E representation
                output_mixed_exp_map = self.ball.expmap0(output_mixed)  # project on the hyperbolic
                normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
                preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
                loss8 = self.loss8(preds_mix, y_a, y_b, lambd)

                loss += (loss6.item() + 2 * loss8.item()) * y.shape[0]

        return loss, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model.to(self.device)
        self.model.eval()
        test_acc = 0.
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                test_num += y.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()

                output = self.model.base(x)
                output_exp_map = self.ball.expmap0(output)
                output_exp_map_norm = F.normalize(output_exp_map, dim=-1, p=2)
                output = self.model.predictor(output_exp_map_norm).float()
                pred = output.max(1, keepdim=True)[1]
                test_acc += pred.eq(y.view_as(pred)).sum().item()

                y_prob.append(output.cpu().numpy())
                y_true.append(label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes)))

        return test_acc, test_num, 0, 0


    def fine_tune(self, which_module=['base', 'predictor']):
        trainloader = self.load_train_data()
        self.model.train()
        module_train_(self.model.base, 'base' in which_module)
        module_train_(self.model.predictor, 'predictor' in which_module)
        # for step in range(self.fine_tuning_steps):
        for step in range(1):
            for i, (x_pos, y, y_neg) in enumerate(trainloader):
                x_pos = x_pos.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                y_exp_map = self.polars[y]  # note refer to prototype
                output_pos = self.model.base(x_pos)  # note E representation
                output_pos_exp_map = self.ball.expmap0(output_pos)  # project on the hyperbolic

                y_neg_exp_map = self.polars[y_neg]
                loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()

                mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
                output_mixed = self.model.base(mixed_x)  # note E representation
                output_mixed_exp_map = self.ball.expmap0(output_mixed)  # project on the hyperbolic
                normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
                preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
                loss8 = self.loss8(preds_mix, y_a, y_b, lambd)

                loss = loss6 + 2 * loss8

                loss.backward()

                self.optimizer.step()