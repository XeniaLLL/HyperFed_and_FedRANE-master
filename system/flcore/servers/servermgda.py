import os
import sys
import copy
from flcore.clients.clientmgda import ClientMGDA
from flcore.servers.serverbase import Server

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from geoopt import PoincareBall
import wandb
from utils.data_utils import read_global_test_data
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.autograd import Variable


class MGDA(Server):
    def __init__(self, args, times):
        # assert args.join_ratio == 1., "All clients are supposed to join training"
        super(MGDA, self).__init__(args, times)

        self.set_slow_clients()
        self.set_clients(args, ClientMGDA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []  # store for time consumption
        self.num_classes = args.num_classes
        self.args = args

        self.grads = dict()
        self.sample_weights = dict()
        self.scale = dict()

    def train(self):
        self.selected_clients = self.select_clients()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:
                print(f"\n---------------------Round number: {i} --------------")
                print("\nEvaluate global model")
                self.evaluate(global_test=True)

            for client in self.selected_clients:
                client.train()

            self.calculate_weight()
            self.aggregate_parameters()

            self.send_models()
            self.Budget.append(time.time() - s_t)
            print("-" * 25, 'time cost', '-' * 25, self.Budget[-1])

        for i in range(self.args.fine_tuning_steps):
            for client in self.clients:
                client.fine_tune()
            print("\n-------------Evaluate fine-tuned model-------------")
            self.evaluate()

    def calculate_weight(self):
        tot_samples = 0
        for client in self.selected_clients:
            self.grads[client.id] = []
            self.sample_weights[client.id] = client.train_samples
            tot_samples += client.train_samples
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                if client_param.grad is not None:
                    # self.grads[client.id].append(server_param.data.clone() - client_param.data.clone())
                    self.grads[client.id].append(client_param.data.clone() - server_param.data.clone())
        # for client in self.selected_clients:
        #     self.grads[client.id] = client.get_gradient()

        gn = {}
        for t in self.grads:
            gn[t] = np.sqrt(sum([gr.pow(2).sum().data.cpu() for gr in self.grads[t]]))
        for client in self.selected_clients:
            for gr_i in range(len(self.grads[client.id])):
                self.grads[client.id][gr_i] = self.grads[client.id][gr_i] / gn[client.id]

        sol, min_norm = MinNormSolver.find_min_norm_element(
            [self.grads[client.id] for client in self.selected_clients],
            sample_weights=[self.sample_weights[client.id] / tot_samples for client in self.selected_clients])
        for i, client in enumerate(self.selected_clients):
            self.scale[client.id] = float(sol[i])

    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.selected_clients[0].model)
        for param in self.global_model.parameters():
            param.data.zero_()

        for client in self.selected_clients:
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                server_param.data += self.scale[client.id] * client_param.data.clone()

    def evaluate(self, acc=None, loss=None, global_test=False):
        if global_test:
            test_data1, test_data2 = read_global_test_data(self.dataset)
            data_loader = DataLoader(test_data2, self.batch_size, drop_last=False, shuffle=True)
            self.global_model.eval()
            test_num, test_acc = 0., 0.
            with torch.no_grad():
                for x, y in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.global_model.base(x)
                    output_exp_map = PoincareBall(self.args.curvature).expmap0(output)  # project on the hyperbolic
                    output_exp_map_norm = F.normalize(output_exp_map, dim=-1, p=2)
                    output = self.global_model.predictor(output_exp_map_norm).float()
                    pred = output.max(1, keepdim=True)[1]

                    test_acc += pred.eq(y.view_as(pred)).sum().item()
                    test_num += y.shape[0]

            global_test2_acc = test_acc / test_num
            if global_test2_acc > self.best_global_test2_acc:
                self.best_global_test2_acc = global_test2_acc
            print("Global test2 acc: {:.4f}".format(global_test2_acc))

        stats = self.test_metrics()
        # sys.exit()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(train_loss)

        if test_acc > self.best_test_accuracy:
            self.best_test_accuracy = test_acc

        if not self.debug:
            if not global_test:
                wandb.log({"train_loss": train_loss,
                           "test_accuracy": test_acc,
                           "best_test_accuracy": self.best_test_accuracy})
            else:
                wandb.log({"train_loss": train_loss,
                           "test_accuracy": test_acc,
                           "best_test_accuracy": self.best_test_accuracy,
                           "global_test2_accuracy": global_test2_acc,
                           "best_global_test2_accuracy": self.best_global_test2_acc,
                           })
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))