from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(DIRNAME.parent))

import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data_for_CL,read_client_data_for_CL_y, read_client_data_for_CL_aug, read_client_data
from flcore.clients.clientbase import Client

class ClientCL(Client):
    def __init__(self,  args, id, train_samples, test_samples, **kwargs):
        super(ClientCL, self).__init__( args, id, train_samples, test_samples, **kwargs)


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data_for_CL(self.dataset, self.id, is_train=True)
        # train_data = read_client_data_for_CL_y(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data_for_CL(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

class ClientCLAug(Client):
    def __init__(self,  args, id, train_samples, test_samples, **kwargs):
        super(ClientCLAug, self).__init__( args, id, train_samples, test_samples, **kwargs)


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data_for_CL_aug(self.dataset,self.transforms, self.id, is_train=True)
        # train_data = read_client_data_for_CL_y(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

class ClientCLY(Client):
    def __init__(self,  args, id, train_samples, test_samples, **kwargs):
        super(ClientCLY, self).__init__( args, id, train_samples, test_samples, **kwargs)


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data_for_CL_y(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data_for_CL_y(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)


#
# if __name__ == '__main__':
#     data= read_client_data_for_CL("Cifar10", '0', True)
#     pass