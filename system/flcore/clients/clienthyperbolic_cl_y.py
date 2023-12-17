from collections import defaultdict
import copy

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time

import wandb
from matplotlib import pyplot as plt

from flcore.clients.clientbase import Client
from flcore.clients.clientbase_cl import ClientCL, ClientCLY
from utils.privacy import *
from flcore.losses.btLoss import AULoss, MixupLoss, mixup_data
from flcore.losses.costripletLoss import CosTripletLoss, InfoNCE, ArcCoshLoss, ACoshTripletLoss
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from geoopt import PoincareBall
import geoopt


def module_train_(module: nn.Module, mode: bool = True):
    module.train(mode)
    for param in module.parameters():
        param.requires_grad_(mode)


class clientHyperbolicCLY(ClientCLY):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super(clientHyperbolicCLY, self).__init__(args, id, train_samples, test_samples, **kwargs)
        self.ball = PoincareBall(args.curvature)
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss6 = ACoshTripletLoss(manifold=self.ball, margin=args.margin_triplet)
        self.loss8 = MixupLoss(alpha=0.5)
        self.optimizer = geoopt.optim.RiemannianSGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                                    weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=750,
                                                         gamma=0.1)
        self.num_classes = args.num_classes
        self.dims = args.HyperbolicFed_dim
        self.one_hot = torch.eye(self.num_classes)
        self.polars = args.predictor.weight.data  # args.classpolars
        self.c = args.curvature
        self.fine_tuning_steps = args.fine_tuning_steps


        if self.test_pm:
            self.sample_per_class_total = torch.zeros(self.num_classes)
            trainloader = self.load_train_data()
            for x, y, y_neg in trainloader:
                for yy in y:
                    self.sample_per_class_total[yy.item()] += 1
            testloader = self.load_test_data()
            for x, y in testloader:
                for yy in y:
                    self.sample_per_class_total[yy.item()] += 1
            self.sample_per_class_total = self.sample_per_class_total / torch.sum(self.sample_per_class_total)

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self):
        # for g in self.optimizer.param_groups:
        #     g['lr'] = self.learning_rate

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
                if type(x_pos) == type([]):
                    x_pos[0] = x_pos[0].to(self.device)
                else:
                    x_pos = x_pos.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()

                y_exp_map = self.polars[y]  # note refer to prototype
                output_pos = self.model.base(x_pos)  # note E representation
                output_pos_exp_map = self.ball.expmap0(output_pos)  # project on the hyperbolic


                # # loss 6
                y_neg_exp_map = self.polars[y_neg]
                loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()


                # # loss8
                mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
                output_mixed = self.model.base(mixed_x)  # note E representation
                output_mixed_exp_map = self.ball.expmap0(output_mixed)  # project on the hyperbolic
                normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
                preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
                loss8 = self.loss8(preds_mix, y_a, y_b, lambd)


                loss = loss6 + 2 * loss8  # careful


                loss.backward()
                if self.privacy:
                    dp_step(self.optimizer, i, len(trainloader))
                else:
                    # self.optimizer.step(self.global_params, self.device)
                    self.optimizer.step()

                # avgloss += loss.item()
                # avglosscount += 1
                # newloss = avgloss / avglosscount

                # output = self.model.predictor(output_exp_map).float()
                # pred = output.max(1, keepdim=True)[1]
                # acc += pred.eq(y.view_as(pred)).sum().item()
            self.scheduler.step()
        # trainlen = len(trainloader.dataset)  # todo check whether verbose for local training
        # newacc = acc / float(trainlen)
        # self.model.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        loss = 0
        train_acc = 0.
        for i, (x_pos, y, y_neg) in enumerate(trainloader):
            if type(x_pos) == type([]):
                x_pos[0] = x_pos[0].to(self.device)
            else:
                x_pos = x_pos.to(self.device)
            train_num += y.shape[0]
            y = y.to(self.device)

            y_exp_map = self.polars[y]  # note refer to prototype
            output_pos = self.model.base(x_pos)  # note E representation
            # output_pos_exp_map = pmath.expmap0(output_pos, c=self.c)  # project on the hyperbolic
            output_pos_exp_map = self.ball.expmap0(output_pos)  # project on the hyperbolic

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

            # eval acc for train set
            normed_output_pos_exp_map = F.normalize(output_pos_exp_map, p=2, dim=1)
            output = self.model.predictor(normed_output_pos_exp_map).float()
            pred = output.max(1, keepdim=True)[1]
            train_acc += pred.eq(y.view_as(pred)).sum().item()

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        # print("Eval acc in train set:", train_acc / train_num)

        return loss, train_num

    def test_metrics(self):
        # # self.model.to(self.device)
        # self.model.eval()
        # test_acc = 0.
        # test_num = 0
        # p_m_y_sum = 0
        # _, test_data2 = read_global_test_data(self.dataset)
        # data_loader = DataLoader(test_data2, self.batch_size, drop_last=False, shuffle=True)
        # with torch.no_grad():
        #     for x, y in data_loader:
        #         test_num += y.shape[0]
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         output = self.model.base(x)
        #         output_exp_map = self.ball.expmap0(output)
        #         output_exp_map_norm = F.normalize(output_exp_map, dim=-1, p=2)
        #         output = self.model.predictor(output_exp_map_norm).float()
        #         result = torch.argmax(output, dim=1) == y
        #         p_m_y = y.float()
        #         for i in range(len(y)):
        #             p_m_y[i] = self.sample_per_class[int(y[i])]
        #         test_acc += torch.sum(result * p_m_y)
        #         p_m_y_sum += torch.sum(p_m_y)
        #
        # test_acc /= p_m_y_sum
        # # self.model.cpu()
        # return test_acc, test_num, 0, 0

        testloaderfull = self.load_test_data()
        # self.model.to(self.device)
        self.model.eval()
        test_acc = 0.
        test_acc_pm = 0
        test_loss = 0
        test_num = 0
        y_prob = []
        y_true = []
        p_m_y_sum = 0
        output_vectors = []
        gt_labels = []

        with torch.no_grad():
            for x, y in testloaderfull:
                test_num += y.shape[0]
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()

                y_exp_map = self.polars[y]  # note refer to prototype
                output = self.model.base(x)  # note E representation
                # output_exp_map = pmath.expmap0(output, c=self.c)  # project on the hyperbolic
                output_exp_map = self.ball.expmap0(output)  # project on the hyperbolic
                output_exp_map_norm = F.normalize(output_exp_map, dim=-1, p=2)
                output = self.model.predictor(output_exp_map_norm).float()
                pred = output.max(1, keepdim=True)[1]

                # output_exp_map = output_exp_map.repeat(1, self.num_classes).reshape(y.shape[0], self.num_classes,
                #                                                                     self.polars.shape[1])
                # y_exp_map = self.polars.repeat(y.shape[0], 1).reshape(y.shape[0], self.num_classes,
                #                                                            self.polars.shape[1])
                # dist = self.ball.dist(output_exp_map, y_exp_map, dim=-1)
                # pred = dist.min(1, keepdim=True)[1]
                # for visualization
                if self.visualize:
                    output_vectors.extend(output_exp_map.cpu().numpy().tolist())
                    gt_labels.extend(y.cpu().numpy().tolist())

                test_acc += pred.eq(y.view_as(pred)).sum().item()
                if self.test_pm:
                    result = torch.argmax(output, dim=1) == y
                    p_m_y = y.float()
                    for i in range(len(y)):
                        p_m_y[i] = self.sample_per_class_total[int(y[i])]
                    test_acc_pm += torch.sum(result * p_m_y)
                    p_m_y_sum += torch.sum(p_m_y)

                # test_loss += self.loss(output_exp_map, y_exp_map)
                y_prob.append(output.cpu().numpy())
                y_true.append(label_binarize(y.cpu().numpy(), classes=np.arange(self.num_classes)))

        # self.model.cpu()
        # y_prob = np.concatenate(y_prob, axis=0)
        # y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        if self.test_pm:
            test_acc_pm /= p_m_y_sum

        return test_acc, test_num, 0, test_acc_pm

    # def set_parameters(self, model):
    #     for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
    #         global_param.data = new_param.data.clone()
    #         param.data = new_param.data.clone()

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

                # loss5 = self.loss5(output_pos_exp_map, y_exp_map).mean()

                y_neg_exp_map = self.polars[y_neg]
                loss6 = self.loss6(output_pos_exp_map, y_exp_map, y_neg_exp_map).mean()

                mixed_x, y_a, y_b, lambd = mixup_data(x_pos, y, self.loss8.alpha)
                output_mixed = self.model.base(mixed_x)  # note E representation
                output_mixed_exp_map = self.ball.expmap0(output_mixed)  # project on the hyperbolic
                normed_output_mixed_exp_map = F.normalize(output_mixed_exp_map, p=2, dim=1)
                preds_mix = self.model.predictor(normed_output_mixed_exp_map).float()
                loss8 = self.loss8(preds_mix, y_a, y_b, lambd)

                # loss = loss5
                loss = loss6 + 2 * loss8

                loss.backward()

                self.optimizer.step()
