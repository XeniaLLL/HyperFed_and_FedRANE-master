import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, read_global_test_data
from utils.data_utils import GlobalTestDataset


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threshold = args.time_threshold
        self.save_folder_name = args.save_folder_name
        self.save_per_epoch = args.save_per_epoch
        self.checkpoint_name = args.checkpoint_name
        self.top_cnt = 100
        self.test_pm = args.test_pm

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        self.suffix = args.suffix
        self.best_test_accuracy = 0
        self.best_test_accuracy_pm = 0
        self.best_global_test_acc = 0
        self.best_global_test1_acc = 0
        self.best_global_test2_acc = 0
        self.debug = args.debug
        self.global_test_dataset = None  # combined test data from all the clients
        self.global_test_acc = 0

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        # return self.clients
        selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def save_checkpoint(self, epoch_num):
        checkpoint_folder = os.path.join("checkpoints", self.dataset)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if epoch_num % self.save_per_epoch == 0:
            print(f"\nSave checkpoint: {epoch_num}")
            torch.save({
                'epoch_num': epoch_num,
                'global_model_state_dict': self.global_model.state_dict(),
                'client_model_state_dict': [client.model.state_dict() for client in self.selected_clients],
                'optimizer_state_dict': [client.optimizer.state_dict() for client in self.selected_clients]
            }, os.path.join(checkpoint_folder, f"{self.checkpoint_name}.pt"))

    def load_checkpoint(self, epoch_num):
        print(f"\nLoad checkpoint: {epoch_num}")
        checkpoint_folder = os.path.join("checkpoints", self.dataset)
        checkpoint_path = os.path.join(checkpoint_folder, f"{self.checkpoint_name}.pt")
        assert (os.path.exists(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.global_model.eval()
        client_model_list = checkpoint['client_model_state_dict']
        optimizer_list = checkpoint['optimizer_state_dict']
        for i in range(len(self.selected_clients)):
            self.selected_clients[i].model.load_state_dict(client_model_list[i])
            self.selected_clients[i].model.eval()
            self.selected_clients[i].optimizer.load_state_dict(optimizer_list[i])

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times) + self.suffix
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_correct_pm = []
        tot_auc = []
        acc_dict = dict()
        for c in self.selected_clients:
            metrics: tuple = c.test_metrics()
            if len(metrics) == 4:
                ct, ns, auc, client_test_pm = metrics
            else:
                assert len(metrics) == 3
                ct, ns, auc = metrics
                client_test_pm = 0  # HACK some clients don't return `ctg` from their `test_metrics`
            tot_correct.append(ct * 1.0)
            tot_correct_pm.append(client_test_pm * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            acc_dict[c.id] = ct / ns
            # print(f"accuracy of client {c.id}:", ct / ns)

        # for i in range(self.join_clients):
        #     print(f"accuracy of client {i}: {acc_dict[i]}")
        ids = [c.id for c in self.selected_clients]
        return ids, num_samples, tot_correct, tot_auc, tot_correct_pm

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    def get_combined_test_data(self):
        """Collects the test data from all the clients"""

        test_data_dict = dict()
        for client in self.clients:
            test_loader = client.load_test_data()
            for x, y in test_loader:
                for i in range(len(y)):
                    test_data_dict[x[i]] = y[i]
        test_x, test_y = list(test_data_dict.keys()), list(test_data_dict.values())
        dataset = GlobalTestDataset(test_x, test_y)
        self.global_test_dataset = dataset

    def global_test(self):
        """use the global model to test the combined test data."""
        test_num, test_acc = 0., 0.
        data_loader = DataLoader(self.global_test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True)
        self.global_model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                pred = output.max(1, keepdim=True)[1]
                test_acc += pred.eq(y.view_as(pred)).sum().item()
                test_num += y.shape[0]
        self.global_test_acc = test_acc / test_num
        if self.global_test_acc > self.best_global_test_acc:
            self.best_global_test_acc = self.global_test_acc
        # if not self.debug:
        #     wandb.log({'global_test_acc': global_test_acc,
        #                'best_global_test_acc': self.best_global_test_acc})
        print("Global test acc: {:.4f}".format(self.global_test_acc))


    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
