import time
import copy
from typing import Optional
from copy import deepcopy

import cvxpy as cp
import numpy as np
import torch
import torch.linalg
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from scipy.optimize import minimize

from utils.data_utils import read_global_test_data
from flcore.clients.clientSphereGAug import ClientSphereGAug
from flcore.servers.serverbase import Server
from utils.min_norm_solvers_cag import MinNormSolver


class FedRANEAug(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        orthogonal_matrix, _ = torch.linalg.qr(
            torch.randn(self.global_model.predictor.in_features, self.global_model.predictor.in_features))
        self.global_model.predictor.weight.data = orthogonal_matrix[:args.num_classes].to(args.device)

        # self.global_model.predictor.bias.data.zero_()
        self.global_model.predictor.register_parameter('bias', None)

        self.aggregate_all = args.aggregate_all

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, ClientSphereGAug)
        self.args = args
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.visualize: bool = args.visualize
        self.fine_tune_epochs: Optional[int] = args.spherefed_fine_tune_epochs if hasattr(args,
                                                                                          'spherefed_fine_tune_epochs') else None

        self.method = args.multi_task_method
        if self.aggregate_all:
            self.global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=1.0)
        else:
            self.global_optimizer = torch.optim.SGD(self.global_model.base.parameters(), lr=1.0)

        if self.method == "CAG":
            self.cagrad_c = args.cagrad_c
        elif self.method == "Nash":
            num_clients = self.join_clients
            for i in range(self.join_clients):
                if self.clients[i].train_samples < args.batch_size:
                    num_clients -= 1
            self.alpha_param = cp.Variable(shape=(num_clients,), nonneg=True)
            self.prvs_alpha_param = cp.Parameter(
                shape=(num_clients,), value=np.ones(num_clients, dtype=np.float32)
            )
            self.G_param = cp.Parameter(
                shape=(num_clients, num_clients), value=np.eye(num_clients)
            )
            self.normalization_factor_param = cp.Parameter(
                shape=(1,), value=np.array([1.0])
            )
            G_prvs_alpha = self.G_param @ self.prvs_alpha_param
            prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
            self.phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
            self.prvs_alpha = np.ones(num_clients, dtype=np.float32)
            G_alpha = self.G_param @ self.alpha_param
            constraint = []
            for i in range(num_clients):
                constraint.append(
                    -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                    - cp.log(G_alpha[i])
                    <= 0
                )
            obj = cp.Minimize(
                cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
            )
            self.prob = cp.Problem(obj, constraint)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(global_test=True)

            for client in self.selected_clients:
                client.train()

            if self.method == "AVG":
                self.receive_models()
                self.aggregate_parameters()
            else:
                self.update_global()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        # self.save_global_model()
        self.send_models()
        self.evaluate()

        # Fine-tune clients' classifier for personalization
        if self.fine_tune_epochs is not None:
            for i in range(self.fine_tune_epochs):
                for client in self.clients:
                    client.fine_tune(1)
                print("\nEvaluate after fine-tune:")
                self.evaluate()

        if self.visualize:
            np.save(
                '../data/feature_output/spherefed_global_prototypes.npy',
                self.global_model.predictor.weight.cpu().numpy()
            )

    def evaluate(self, acc=None, loss=None, global_test=False):
        if global_test:
            test_data1, test_data2 = read_global_test_data(self.dataset)
            data_loader = DataLoader(test_data2, self.batch_size, drop_last=False, shuffle=True)
            self.global_model.to(self.device)
            self.global_model.eval()
            test_num, test_acc = 0., 0.
            with torch.no_grad():
                for x, y in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    z = F.normalize(self.global_model.base(x))
                    output = self.global_model.predictor(z)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]
            global_test2_acc = test_acc / test_num
            if global_test2_acc > self.best_global_test2_acc:
                self.best_global_test2_acc = global_test2_acc
            print("Global test2 acc: {:.4f}".format(global_test2_acc))
            self.global_model.cpu()

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_acc_pm = sum(stats[4]) * 1.0 / self.join_clients
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if test_acc > self.best_test_accuracy:
            self.best_test_accuracy = test_acc

        if self.test_pm and test_acc_pm > self.best_test_accuracy_pm:
            self.best_test_accuracy_pm = test_acc_pm

        if not self.debug:
            if not global_test:
                wandb.log({"train_loss": train_loss,
                           "test_accuracy": test_acc,
                           "best_test_accuracy": self.best_test_accuracy})
            else:
                if not self.test_pm:
                    wandb.log({"train_loss": train_loss,
                               "test_accuracy": test_acc,
                               "best_test_accuracy": self.best_test_accuracy,
                               # "global_test1_accuracy": global_test1_acc,
                               # "best_global_test1_accuracy": self.best_global_test1_acc,
                               "global_test2_accuracy": global_test2_acc,
                               "best_global_test2_accuracy": self.best_global_test2_acc,
                               })
                else:
                    wandb.log({"train_loss": train_loss,
                               "test_accuracy": test_acc,
                               "best_test_accuracy": self.best_test_accuracy,
                               "test_accuracy_pm": test_acc_pm,
                               "best_test_accuracy_pm": self.best_test_accuracy_pm,
                               # "global_test1_accuracy": global_test1_acc,
                               # "best_global_test1_accuracy": self.best_global_test1_acc,
                               "global_test2_accuracy": global_test2_acc,
                               "best_global_test2_accuracy": self.best_global_test2_acc,
                               })
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def update_global(self):
        self.global_optimizer.zero_grad()
        grad_dims = []
        possible_i = None  # find the first client with non-empty grads
        if self.aggregate_all:
            for i in range(len(self.selected_clients)):
                for param in self.selected_clients[i].model.parameters():
                    if param.grad is not None:
                        grad_dims.append(param.data.numel())
                if len(grad_dims) > 0:
                    possible_i = i
                    break
        else:
            for i in range(len(self.selected_clients)):
                for param in self.selected_clients[i].model.base.parameters():
                    if param.grad is not None:
                        grad_dims.append(param.data.numel())
                if len(grad_dims) > 0:
                    possible_i = i
                    break

        grads = torch.Tensor(sum(grad_dims), self.join_clients)
        no_grad_id_list = []

        for i in range(self.join_clients):
            client = self.selected_clients[i]
            index = i - len(no_grad_id_list)  # handle clients with empty grads
            grads[:, index].fill_(0.0)
            cnt = 0
            if self.aggregate_all:
                for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                    if client_param.grad is not None:
                        grad_cur = server_param.data.detach().clone() - client_param.data.detach().clone()
                        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                        en = sum(grad_dims[:cnt + 1])
                        grads[beg:en, index].copy_(grad_cur.data.view(-1))
                        cnt += 1
            else:
                for server_param, client_param in zip(self.global_model.base.parameters(),
                                                      client.model.base.parameters()):
                    if client_param.grad is not None:
                        grad_cur = server_param.data.detach().clone() - client_param.data.detach().clone()
                        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                        en = sum(grad_dims[:cnt + 1])
                        grads[beg:en, index].copy_(grad_cur.data.view(-1))
                        cnt += 1
            if not grads[:, index].any():
                keep_columns = list(range(0, index)) + list(range(index + 1, grads.shape[1]))
                grads = torch.index_select(grads, dim=1, index=torch.tensor(keep_columns))
                no_grad_id_list.append(client.id)

        if self.method == "CAG":
            new_grads = self.cagrad(grads, no_grad_id_list)  # grads: dim x K clients
        elif self.method == "MGDA":
            new_grads = self.mgda(grads)
        elif self.method == "PCG":
            new_grads = self.pcgrad(grads)
        elif self.method == "Nash":
            new_grads = self.nash(grads, no_grad_id_list)

        self.global_model.train()
        cnt = 0
        if self.aggregate_all:
            for server_param, client_param in zip(self.global_model.parameters(),
                                                  self.selected_clients[possible_i].model.parameters()):
                if client_param.grad is not None:
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    this_grad = new_grads[beg: en].contiguous().view(server_param.data.size())
                    server_param.grad = this_grad.data.clone().to(server_param.device)
                    cnt += 1
        else:
            for server_param, client_param in zip(self.global_model.base.parameters(),
                                                  self.selected_clients[possible_i].model.base.parameters()):
                if client_param.grad is not None:
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    this_grad = new_grads[beg: en].contiguous().view(server_param.data.size())
                    server_param.grad = this_grad.data.clone().to(server_param.device)
                    cnt += 1

        self.global_optimizer.step()

    def cagrad(self, grads, no_grad_id_list):
        grad_vec = grads.t()
        tot_samples = 0
        sample_weights = dict()
        for client in self.selected_clients:
            sample_weights[client.id] = client.train_samples
            tot_samples += client.train_samples

        x_start = np.array([sample_weights[client.id] / tot_samples for client in self.selected_clients if
                            client.id not in no_grad_id_list])
        # x_start = np.ones(self.join_clients) / self.join_clients

        grads = grad_vec / 100.
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.cagrad_c * g0.norm()).cpu().item()

        num_clients = self.join_clients - len(no_grad_id_list)

        def objfn(x):
            return (x.reshape(1, num_clients).dot(A).dot(b.reshape(num_clients, 1)) +
                    c * np.sqrt(
                        x.reshape(1, num_clients).dot(A).dot(x.reshape(num_clients, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100

    def mgda(self, grads):
        tot_samples = 0
        sample_weights = dict()
        for client in self.selected_clients:
            sample_weights[client.id] = client.train_samples
            tot_samples += client.train_samples

        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([grads_cpu[t] for t in range(grads.shape[-1])],
                                                            sample_weights=[sample_weights[client.id] / tot_samples
                                                                            for client in self.selected_clients])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def pcgrad(self, grads):
        rng = np.random.default_rng()
        grad_vec = grads.t()

        shuffled_task_indices = np.zeros((self.join_clients, self.join_clients - 1), dtype=int)
        for i in range(self.join_clients):
            task_indices = np.arange(self.join_clients)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def nash(self, grads, no_grad_id_list):
        def stop_criteria(gtg, alpha_t):
            return (
                    (self.alpha_param.value is None)
                    or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                    or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
            )

        def solve_optimization(gtg, no_grad_id_list):
            self.G_param.value = gtg
            self.normalization_factor_param.value = self.normalization_factor
            tot_samples = 0
            sample_weights = dict()
            for client in self.selected_clients:
                sample_weights[client.id] = client.train_samples
                tot_samples += client.train_samples

            alpha_t = np.array([sample_weights[client.id] / tot_samples for client in self.selected_clients if
                                client.id not in no_grad_id_list])

            for _ in range(20):  # optim_niter
                self.alpha_param.value = alpha_t
                self.prvs_alpha_param.value = alpha_t
                try:
                    self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
                except:
                    self.alpha_param.value = self.prvs_alpha_param.value
                if stop_criteria(gtg, alpha_t):
                    break
                alpha_t = self.alpha_param.value

            if alpha_t is not None:
                self.prvs_alpha = alpha_t
            return self.prvs_alpha

        # grads: dim x num_clients
        GTG = torch.mm(grads.t(), grads)
        self.normalization_factor = (
            torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        )
        GTG = GTG / self.normalization_factor.item()
        alpha = solve_optimization(GTG.cpu().detach().numpy(), no_grad_id_list)
        w = torch.FloatTensor(alpha).to(grads.device)

        w /= sum(alpha)

        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def add_parameters(self, w, client_model):
        if self.aggregate_all:
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w
        else:
            for server_param, client_param in zip(self.global_model.base.parameters(), client_model.base.parameters()):
                server_param.data += client_param.data.clone() * w
            for server_param, client_param in zip(self.global_model.predictor.parameters(), client_model.predictor.parameters()):
                server_param.data += client_param.data.clone() * w
