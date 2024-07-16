import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch.nn.functional as F

from flcore.clients.clientbase_cl import ClientCLAug
from flcore.losses.btLoss import AULoss, MixupLoss, mixup_data
from flcore.optimizers.fedoptimizer import SAM, ASAM
from torchvision import transforms
import random
from PIL import ImageFilter

# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
#     Taken from https://github.com/facebookresearch/moco/blob/master/moco/loader.py
#     """
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x

class ClientSphereGAug(ClientCLAug):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.MSELoss()  # to bypass the scaling issue of CE
        # self.loss = nn.CosineSimilarity(eps=1e-9)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5,
                                         momentum=0.9)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=args.global_rounds+1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=150 * self.local_steps,
                                                         gamma=0.1)

        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                              saturation=0.8, hue=0.2)
        img_size= 32 if ('cifar' in args.dataset.lower()) else 28
        self.transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(sigma=[0.1, 2.0], kernel_size=3),
                                              ])

        self.loss8 = MixupLoss(alpha=0.5)
        self.alpha = args.reg_graph_aug
        self.use_sam = args.use_sam
        self.rho = args.sam_rho
        self.eta = args.sam_eta
        self.temperature = args.info_nce_temperature

        self.L_ce = nn.CrossEntropyLoss()
        self.aggregate_all = args.aggregate_all
        self.nce_reg = args.nce_reg
        self.trainloader= self.load_train_data()
        self.testloader= self.load_test_data()

        # fix the classifier
        for param in self.model.predictor.parameters():
            param.requires_grad = False

    def feature_extract(self, x, y, is_train=True):
        """ Extract feature vector for the TRAIN data.

        Returns:
            A tuple (y, z), where y is the one_hot vector and z is the normalized feature vector

        """
        x = x.to(self.device)
        y = y.to(self.device)

        y_ = F.one_hot(y.to(torch.int64), self.num_classes).float()  # for MSE loss

        # add mixup
        # mixed_x, y_a, y_b, lambd = mixup_data(x, y, self.loss8.alpha)
        # output_mixed = self.model.base(mixed_x)  # note E representation
        # output_mixed_norm = F.normalize(output_mixed, p=2, dim=1)
        # preds_mix = self.model.predictor(output_mixed_norm).float()
        # loss8 = self.loss8(preds_mix, y_a, y_b, lambd)
        # loss += loss8

        if is_train:
            mixed_x, y_a, y_b, lambd = mixup_data(x, y, self.loss8.alpha)
            # y_a = F.one_hot(y_a.to(torch.int64), self.num_classes).float()  # for MSE loss
            # y_b = F.one_hot(y_b.to(torch.int64), self.num_classes).float()  # for MSE loss

            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            # feature extractor maps input x to a vector on the unit hypersphere
            z = self.model.base(torch.cat((x, mixed_x), dim=0))
            return F.normalize(z), y_, y_a, y_b, lambd
        else:
            with torch.no_grad():
                z = self.model.base(x)
            z = F.normalize(z)
            return y_, z

    def info_nce_loss(self, Z1, Z2, n_views=2):
        N_BS = Z1.shape[0]
        features = torch.cat((Z1, Z2))
        labels = torch.cat([torch.arange(N_BS) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train_mixup(self, mixup_x, y_a, y_b, lambd):
        out_mix = self.model.predictor(mixup_x)
        loss = self.loss8(out_mix, y_a, y_b, lambd)
        return loss

    def train_raw(self):
        # trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x,x_aug, y) in enumerate(self.trainloader):
                y=y.to(self.device)
                self.optimizer.zero_grad()
                # p = self.model.predictor.weight.data[y.to(self.device)]
                z, y_, y_a, y_b, lambd = self.feature_extract(x, y)
                out = z[:x.shape[0]]
                z_mix = z[x.shape[0]:]
                loss = 0

                if self.alpha != 1:   # 1 for not use graph
                    A = self.model.graph_generator.get_graph(out)
                    Z1_aug = self.model.gnn(out, A)
                    Z1_aug = F.normalize(Z1_aug, dim=-1)
                    logits, label = self.info_nce_loss(out, Z1_aug, 2)
                    loss = self.nce_reg * self.L_ce(logits, label)
                    out = self.alpha * out + (1 - self.alpha) * Z1_aug  # careful

                    # mixup
                    A_mix = self.model.graph_generator.get_graph(z_mix)
                    Z1_aug_mix = self.model.gnn(z_mix, A_mix)
                    Z1_aug_mix = F.normalize(Z1_aug_mix, dim=-1)
                    logits_mix, label_mix = self.info_nce_loss(z_mix, Z1_aug_mix, 2)
                    loss+= self.nce_reg * self.L_ce(logits_mix, label_mix)
                    z_mix = self.alpha * z_mix + (1 - self.alpha) * Z1_aug_mix  # careful

                out  = self.model.predictor(out)
                loss+= self.L_ce(out, y)
                loss+= 10*(lambd*self.L_ce(z_mix, y_a)+ (1-lambd)* self.L_ce(z_mix, y_b))
                # loss +=100* self.loss(out, y_)  # ||wz - one_hot(y)||^2]

                # loss = (1 - self.loss(z, p)).pow(2).sum()
                # loss += self.train_mixup(z_mix, y_a, y_b, lambd)
                # print('no mixup')
                loss.backward()
                self.optimizer.step()
        self.scheduler.step()

        self.model.cpu()

    def train_graph(self):
        # trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x,x_aug, y) in enumerate(self.trainloader):
                # p = self.model.predictor.weight.data[y.to(self.device)]
                x_cat= torch.cat((x,x_aug))
                y= torch.cat((y,y))
                z, y, y_a, y_b, lambd = self.feature_extract(x_cat, y)
                out = z[:x_cat.shape[0]]
                z_mix = z[x_cat.shape[0]:]
                loss = 0
                if self.alpha != 1:   # 1 for not use graph
                    A = self.model.graph_generator.get_graph(out)
                    Z1_aug = self.model.gnn(out, A)
                    Z1_aug = F.normalize(Z1_aug, dim=-1)
                    # cl loss (z1,z2)
                    z1, z2= Z1_aug[:x.shape[0]], Z1_aug[x.shape[0]:]
                    logits_cl, label_cl = self.info_nce_loss(z1, z2, 2)
                    loss += 10*self.nce_reg * self.L_ce(logits_cl, label_cl)

                    # cd loss (z, z_aug)
                    # logits, label = self.info_nce_loss(out, Z1_aug, 2)
                    # loss += self.nce_reg * self.L_ce(logits, label)
                    out = self.alpha * out + (1 - self.alpha) * Z1_aug  # careful

                out = self.model.predictor(out)
                loss += self.loss(out, y)  # ||wz - one_hot(y)||^2]

                # loss = (1 - self.loss(z, p)).pow(2).sum()
                # loss += self.train_mixup(z_mix, y_a, y_b, lambd)
                loss.backward()
                self.optimizer.step()
        self.scheduler.step()

        self.model.cpu()

    def train_sam(self):
        # trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):

            # minimizer = SAM(self.optimizer, self.model, self.rho, self.eta)
            minimizer = ASAM(self.optimizer, self.model, self.rho, self.eta)
            for i, (x, y) in enumerate(self.trainloader):
                # p = self.model.predictor.weight.data[y.to(self.device)]
                z, y_, y_a, y_b, lambd = self.feature_extract(x, y)
                out = z[:x.shape[0]]
                z_mix = z[x.shape[0]:]
                loss = 0
                if self.alpha != 1:   # if use graph
                    A = self.model.graph_generator.get_graph(out)
                    Z1_aug = self.model.gnn(out, A)
                    Z1_aug = F.normalize(Z1_aug, dim=-1)
                    logits, label = self.info_nce_loss(out, Z1_aug, 2)
                    loss = self.nce_reg * self.L_ce(logits, label)
                    out = self.alpha * out + (1 - self.alpha) * Z1_aug  # careful

                out = self.model.predictor(out)
                loss += self.loss(out, y_)  # ||wz - one_hot(y)||^2]
                loss += self.train_mixup(z_mix, y_a, y_b, lambd)
                loss.backward()
                minimizer.ascent_step()

                z, y_, y_a, y_b, lambd = self.feature_extract(x, y)
                out = z[:x.shape[0]]
                z_mix = z[x.shape[0]:]
                loss = 0
                if self.alpha != 1:   # if use graph
                    A = self.model.graph_generator.get_graph(out)
                    Z1_aug = self.model.gnn(out, A)
                    Z1_aug = F.normalize(Z1_aug, dim=-1)
                    logits, label = self.info_nce_loss(out, Z1_aug, 2)
                    loss = self.nce_reg * self.L_ce(logits, label)
                    out = self.alpha * out + (1 - self.alpha) * Z1_aug  # careful
                out = self.model.predictor(out)
                loss += self.loss(out, y_)  # ||wz - one_hot(y)||^2]
                # loss+=0.1 * self.info_nce_loss(out, Z1_aug,2)
                loss += self.train_mixup(z_mix, y_a, y_b, lambd)
                loss.backward()
                minimizer.descent_step()
        self.scheduler.step()

        self.model.cpu()

    def train(self):
        if not self.use_sam:
            self.train_graph()
        else:
            self.train_sam()

    def ffc_compute(self):
        """ Compute V and U for the Fast Federated Calibration algorithm."""

        # trainloader = self.load_train_data()

        self.model.to(self.device)
        self.model.eval()
        v, u = 0, 0
        for i, (x, y) in enumerate(self.trainloader):
            y, z = self.feature_extract(x, y, is_train=False)
            # dimension of y is [batch_size, num_classes], z is [batch_size, feature_dimension]
            v += (z.unsqueeze(2) @ z.unsqueeze(1)).sum(dim=0)  # \sum z_i^T z_i
            u += (z.unsqueeze(2) @ y.unsqueeze(1)).sum(dim=0)  # \sum z_i^T one_hot(y)
        self.model.cpu()
        return v, u

    def fine_tune(self, epochs: int):
        # train_loader = self.load_train_data()
        self.model.to(self.device)
        # freeze base
        self.model.base.eval()
        for param in self.model.base.parameters():
            param.requires_grad = False
        # unfreeze predictor
        self.model.predictor.eval()
        for param in self.model.predictor.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(
            self.model.predictor.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            momentum=0.9
        )
        for _ in range(epochs):
            for x, y in self.train_loader:
                y, z = self.feature_extract(x, y, is_train=False)
                output = self.model.predictor(z)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.cpu()

    def train_metrics(self):
        # trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()
        train_num = 0
        loss = 0
        for x,_, y in self.trainloader:
            # p = self.model.predictor.weight.data[y.to(self.device)]
            y, z = self.feature_extract(x, y, is_train=False)
            output = self.model.predictor(z)
            train_num += y.shape[0]
            # loss += (1 - self.loss(z, p)).pow(2).sum()
            loss += self.loss(output, y).item() * y.shape[0]
        self.model.cpu()
        return loss, train_num

    def test_metrics(self):
        # testloaderfull = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()
        test_acc = 0
        test_num = 0
        p_m_y_sum = 0
        test_acc_pm = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in self.testloader:
                if len(y) == 1:
                    continue
                x = x.to(self.device)
                y = y.to(self.device)
                z = F.normalize(self.model.base(x))
                output = self.model.predictor(z)
                result = torch.argmax(output, dim=1) == y
                test_acc += torch.sum(result).item()
                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))
        self.model.cpu()
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        if self.test_pm:
            test_acc_pm /= p_m_y_sum
        return test_acc, test_num, auc, test_acc_pm

    def save_features(self):
        loader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()
        features = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                z = F.normalize(self.model.base(x))
                features.append(z.cpu().numpy())
        self.model.cpu()
        features = np.concatenate(features)
        np.save(f'../data/feature_output/spherefed_local_{self.id}_features.npy', features)

    def set_parameters(self, model):
        if self.aggregate_all:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
