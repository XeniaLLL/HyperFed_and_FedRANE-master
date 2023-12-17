import copy
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

from flcore.clients.clienthyperbolic import clientHyperbolic
from flcore.clients.clienthyperbolic_cl import clientHyperbolicCL
from flcore.clients.clienthyperbolic_cl_y import clientHyperbolicCLY
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from flcore.losses.MMA import get_mma_loss
from geoopt import PoincareBall
import wandb
from utils.data_utils import read_global_test_data
from openTSNE import TSNE
import seaborn as sns


class HyperbolicFed(Server):
    def __init__(self, args, times):
        # assert args.join_ratio == 1., "All clients are supposed to join training"
        super(HyperbolicFed, self).__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientHyperbolicCLY)
        # self.set_clients(args, clientHyperbolic)
        # self.get_combined_test_data()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []  # store for time consumption
        self.num_classes = args.num_classes
        self.args = args
        self.polars = args.predictor.weight.data
        self.global_visual = args.visualize
        self.client_visual = args.visualize

    # def set_clients(self, args, clientObj):
    #     super(SphereFed, self).set_clients(args, clientObj)

    def train(self):
        self.selected_clients = self.select_clients()

        # checkpoint_path = os.path.join(f"checkpoints/{self.dataset}", f"{self.checkpoint_name}.pt")
        # assert (os.path.exists(checkpoint_path))
        # checkpoint = torch.load(checkpoint_path)
        # self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        # self.global_model.eval()
        # client_model_list = checkpoint['client_model_state_dict']
        # optimizer_list = checkpoint['optimizer_state_dict']
        # for i in range(len(self.selected_clients)):
        #     self.selected_clients[i].model.load_state_dict(client_model_list[i])
        #     self.selected_clients[i].model.eval()
        #     self.selected_clients[i].optimizer.load_state_dict(optimizer_list[i])

        for i in range(self.global_rounds + 1):
            # if i == 150:
            #     for client in self.clients:
            #         client.learning_rate = self.learning_rate / 10

            # while not self.done:
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:
                print(f"\n---------------------Round number: {i} --------------")
                print("\nEvaluate global model")
                # self.global_test()
                if i % 20 == 0 and self.args.visualize:
                    self.client_visual = True
                else:
                    self.client_visual = False
                self.evaluate(global_test=True)
                # sys.exit()

            # self.save_checkpoint(i)

            for client in self.selected_clients:
                client.visualize = self.client_visual
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            # self.compute_closed_form_opt_W()
            # self.update_global_classifier(r=1, model=self.global_model)  # todo recheck urgent r
            self.send_models()  # 同fedavg
            self.Budget.append(time.time() - s_t)
            print("-" * 25, 'time cost', '-' * 25, self.Budget[-1])
            # self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt) # div_value=None by default

        torch.save({
            'global_model_state_dict': self.global_model.state_dict(),
            'client_model_state_dict': [client.model.state_dict() for client in self.selected_clients],
            'optimizer_state_dict': [client.optimizer.state_dict() for client in self.selected_clients]
        }, os.path.join(f"checkpoints/{self.dataset}", f"{self.checkpoint_name}.pt"))

        for i in range(self.args.fine_tuning_steps):
            for client in self.clients:
                client.fine_tune()
            print("\n-------------Evaluate fine-tuned model-------------")
            self.evaluate()

        torch.save({
            'global_model_state_dict': self.global_model.state_dict(),
            'client_model_state_dict': [client.model.state_dict() for client in self.selected_clients],
            'optimizer_state_dict': [client.optimizer.state_dict() for client in self.selected_clients]
        }, os.path.join(f"checkpoints/{self.dataset}", f"{self.checkpoint_name}_finetune.pt"))

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def evaluate(self, acc=None, loss=None, global_test=False):
        output_vectors = []
        gt_labels = []
        if global_test:
            test_data1, test_data2 = read_global_test_data(self.dataset)
            data_loader = DataLoader(test_data1, self.batch_size, drop_last=False, shuffle=True)
            # self.global_model.to(self.device)
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
                    # output_exp_map = output_exp_map.repeat(1, self.num_classes).reshape(y.shape[0], self.num_classes,
                    #                                                                     self.polars.shape[1])
                    # y_exp_map = self.polars.repeat(y.shape[0], 1).reshape(y.shape[0], self.num_classes,
                    #                                                       self.polars.shape[1])
                    # dist = PoincareBall(self.args.curvature).dist(output_exp_map, y_exp_map, dim=-1)
                    # pred = dist.min(1, keepdim=True)[1]
                    test_acc += pred.eq(y.view_as(pred)).sum().item()
                    test_num += y.shape[0]
            global_test1_acc = test_acc / test_num
            if global_test1_acc > self.best_global_test1_acc:
                self.best_global_test1_acc = global_test1_acc
            print("Global test1 acc: {:.4f}".format(global_test1_acc))

            data_loader = DataLoader(test_data2, self.batch_size, drop_last=False, shuffle=True)
            self.global_model.eval()
            test_num, test_acc = 0., 0.
            with torch.no_grad():
                for x, y in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.global_model.base(x)
                    output_exp_map = PoincareBall(self.args.curvature).expmap0(output)  # project on the hyperbolic
                    if self.args.visualize:
                        output_vectors.extend(output_exp_map.cpu().numpy().tolist())
                        gt_labels.extend(y.cpu().numpy().tolist())

                    output_exp_map_norm = F.normalize(output_exp_map, dim=-1, p=2)
                    output = self.global_model.predictor(output_exp_map_norm).float()
                    pred = output.max(1, keepdim=True)[1]
                    # output_exp_map = output_exp_map.repeat(1, self.num_classes).reshape(y.shape[0], self.num_classes,
                    #                                                                     self.polars.shape[1])
                    # y_exp_map = self.polars.repeat(y.shape[0], 1).reshape(y.shape[0], self.num_classes,
                    #                                                       self.polars.shape[1])
                    # dist = PoincareBall(self.args.curvature).dist(output_exp_map, y_exp_map, dim=-1)
                    # pred = dist.min(1, keepdim=True)[1]
                    test_acc += pred.eq(y.view_as(pred)).sum().item()
                    test_num += y.shape[0]

            global_test2_acc = test_acc / test_num
            if global_test2_acc > self.best_global_test2_acc:
                self.best_global_test2_acc = global_test2_acc
                if self.global_visual:
                    # output_vectors = np.array(output_vectors)
                    output_vectors = np.concatenate((output_vectors, self.polars.cpu().numpy()))
                    gt_labels = np.array(gt_labels).reshape(-1, 1)
                    # subset_index = np.random.choice(np.array(range(0, gt_labels.shape[0])), 100, replace=False)
                    # output_vectors_small = output_vectors[subset_index, :]
                    # gt_labels_small = gt_labels[subset_index]
                    # prototypes = self.polars.clone().cpu()
                    # visualize_learned_emb(dims=self.dims, num_classes=self.num_classes, prototypes=prototypes,
                    #                       gt_labels_small=gt_labels_small,
                    #                       output_vectors_small=output_vectors_small, is_debug=self.debug, data_id=self.id)
                    embed = TSNE(n_jobs=4).fit(output_vectors)
                    pd_embed = pd.DataFrame(embed)
                    pd_embed_prototype = pd_embed[len(gt_labels):]
                    pd_embed_prototype.insert(loc=2, column='class ID', value=range(self.num_classes))
                    pd_embed_data = pd_embed[:len(gt_labels)]
                    pd_embed_data.insert(loc=2, column='label', value=gt_labels)
                    sns.set_context({'figure.figsize': [15, 10]})
                    color_dict = {0: "#1f77b4",  # 1f77b4
                                  1: "#ff7f0e",  # ff7f0e
                                  2: '#2ca02c',  # 2ca02c
                                  3: '#d62728',  # d62728
                                  4: '#9467bd',  # 9467bd
                                  5: '#8c564b',  # 8c564b
                                  6: '#e377c2',  # e377c2
                                  7: '#7f7f7f',  # 7f7f7f
                                  8: '#bcbd22',  # bcbd22
                                  9: '#17becf'}  # 17becf
                    sns.scatterplot(x=0, y=1, hue="label", data=pd_embed_data, legend=False,
                                    palette=color_dict)
                    sns.scatterplot(x=0, y=1, hue="class ID", data=pd_embed_prototype, s=200,
                                    palette=color_dict)
                    plt.axis('off')
                    # plt.show()
                    plt.savefig(f'tSNE/Cifar10alpha05/server.png', dpi=300)
                    plt.close()
            print("Global test2 acc: {:.4f}".format(global_test2_acc))
            # self.global_model.cpu()

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_acc = sum(stats[2]) * 1.0 / self.join_clients
        test_acc_pm = sum(stats[4]) * 1.0 / self.join_clients
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(train_loss)

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
                               "global_test1_accuracy": global_test1_acc,
                               "best_global_test1_accuracy": self.best_global_test1_acc,
                               "global_test2_accuracy": global_test2_acc,
                               "best_global_test2_accuracy": self.best_global_test2_acc,
                               })
                else:
                    wandb.log({"train_loss": train_loss,
                               "test_accuracy": test_acc,
                               "best_test_accuracy": self.best_test_accuracy,
                               "test_accuracy_pm": test_acc_pm,
                               "best_test_accuracy_pm": self.best_test_accuracy_pm,
                               "global_test1_accuracy": global_test1_acc,
                               "best_global_test1_accuracy": self.best_global_test1_acc,
                               "global_test2_accuracy": global_test2_acc,
                               "best_global_test2_accuracy": self.best_global_test2_acc,
                               })
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        if self.test_pm:
            print("Averaged Test Accuracy with PM: {:.4f}".format(test_acc_pm))
        # print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))
