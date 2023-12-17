'''
Contains the convolutional autoencoder (AE) class with residual architecture for CIFAR datasets
Author: Hadi Jamali-Rad
e-mail: h.jamali.rad@gmail.com
'''
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Contains the convolutional autoencode (AE) models for CIFAR10 dataset
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import flcore.trainmodel.umap_model


# FLT encoder
class Encoder():
    '''
    Encdoer for clustering
    '''

    def __init__(self, ae_model, ae_model_name, model_root_dir, manifold_dim, dataset, client_name, dataset_name="",
                 train_umap=False, use_AE=False, device="cpu"):
        self.ae_model = ae_model
        self.ae_model_name = ae_model_name
        self.model_root_dir = model_root_dir
        self.manifold_dim = manifold_dim
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.client_name = client_name
        self.train_umap = train_umap
        self.use_AE = use_AE
        self.device = device

    def autoencoder(self):
        # get the embedding and corresponding labels
        if self.use_AE:
            embedding_list = []
            labels_list = []
            with torch.no_grad():
                for x, y in self.dataset:
                    x = x.to(self.device)
                    labels_list.append(y)
                    _, embedding = self.ae_model(x.unsqueeze(0))
                    embedding_list.append(embedding.cpu().detach().numpy())
            self.ae_embedding_np = np.concatenate(embedding_list, axis=0)
            self.as_labels_np = np.array(labels_list)

    def manifold_approximation_umap(self):
        # check if manifold approximation is needed
        if self.use_AE and self.manifold_dim == self.ae_embedding_np.shape[1]:
            raise AssertionError('We dont need manifold learning, AE dim =2!')

        # dataset
        data_list = [data[0] for data in self.dataset]  # 迭代拿到x
        data_tensor = torch.cat(data_list, dim=0)
        data_2D_np = torch.reshape(data_tensor, (data_tensor.shape[0], -1)).numpy()
        labels_np = np.array([data[1] for data in self.dataset])

        if self.use_AE:
            if (labels_np != self.ae_labels_np).all():
                raise AssertionError("Order of data samples is shuffled")
            print('Using AE for E2E encoding ...')
            umap_data = self.ae_embedding_np
            umap_model_address = f'{self.model_root_dir}/umap_reducer_{self.dataset_name}_{self.ae_model_name}.p'
        else:
            print('AE not used in this scenario ...')
            umap_data = data_2D_np
            umap_model_address = f'{self.model_root_dir}/umap_reducer_{self.dataset_name}.p'

        # careful place for clustering in the manifold
        if self.train_umap:
            print("Training UMAP on AE embedding ...")
            self.umap_reducer = umap.UMAP(n_components=self.manifold_dim, random_state=42)
            self.umap_embedding = self.umap_reducer.fit_transform(umap_data)
            print(f"UMAP embeding for client_{self.client_name} is extracted. ")
        else:
            self.umap_reducer = pickle.load(open(umap_model_address, 'rb'))
            print("Loading UMAP embedding...")
            self.umap_embedding = self.umap_reducer.transform(umap_data)
            print(f'UMAP embedding/reducer for client_{self.client_name} is loaded')


# latent_size = 256

# batch norm is not a common proactice in AE's
class ConvAutoencoderCIFAR(nn.Module):
    def __init__(self, latent_size):
        super(ConvAutoencoderCIFAR, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # conv layer (depth from 32 --> 4), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # dense layers
        self.fc1 = nn.Linear(8 * 8 * 4,
                             latent_size)  # flattening (input should be calculated by a forward pass - stupidity of Pytorch)

        ## decoder layers ##
        # decoding dense layer
        self.dec_linear_1 = nn.Linear(latent_size, 8 * 8 * 4)
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 32, 1, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x, return_comp=True):
        ## ==== encode ==== ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv3(x))
        # x = self.pool(x)
        # flatten and apply dense layer
        x = x.view(-1, 8 * 8 * 4)
        x_comp = self.fc1(x)  # compressed layer

        ## ==== decode ==== ##
        x = self.dec_linear_1(x_comp)
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x.view(-1, 4, 8, 8)))
        x = F.relu(self.t_conv2(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv3(x))

        if return_comp:
            return x, x_comp
        else:
            return x


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()

        self.num_hiddens = num_hiddens

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #
        self._batchnorm_1 = nn.BatchNorm2d(num_hiddens // 2)
        #
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        #
        self._batchnorm_2 = nn.BatchNorm2d(num_hiddens)
        #
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        #
        self._batchnorm_3 = nn.BatchNorm2d(num_hiddens)
        #
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_4 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens // 2,
                                 kernel_size=1, stride=1)
        #
        self._batchnorm_4 = nn.BatchNorm2d(num_hiddens // 2)
        #
        self._conv_5 = nn.Conv2d(in_channels=num_hiddens // 2, out_channels=num_hiddens // 16,
                                 kernel_size=1, stride=1)
        #
        self._batchnorm_5 = nn.BatchNorm2d(num_hiddens // 16)
        #
        self.fc1 = nn.Linear(8 * 8 * num_hiddens // 16, embedding_dim)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._batchnorm_1(x)

        x = self._conv_2(x)
        x = F.relu(x)
        x = self._batchnorm_2(x)

        x = self._conv_3(x)
        # x = self._batchnorm_3(x)
        x = self._residual_stack(x)

        x = self._conv_4(x)
        x = F.relu(x)
        # x = self._batchnorm_4(x)

        x = self._conv_5(x)
        x = F.relu(x)
        # x = self._batchnorm_5(x)

        x = x.view(-1, 8 * 8 * self.num_hiddens // 16)
        x_comp = self.fc1(x)
        return x_comp


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self.num_hiddens = num_hiddens

        self._linear_1 = nn.Linear(in_channels, 8 * 8 * num_hiddens // 16)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens // 16, out_channels=num_hiddens // 2,
                                                kernel_size=1, stride=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=num_hiddens,
                                                kernel_size=1, stride=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_3 = nn.Conv2d(in_channels=num_hiddens,
                                       out_channels=num_hiddens,
                                       kernel_size=3,
                                       stride=1, padding=1)

        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_5 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._linear_1(inputs)

        x = x.view(-1, self.num_hiddens // 16, 8, 8)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._residual_stack(x)

        x = self._conv_trans_3(x)
        x = F.relu(x)

        x = self._conv_trans_4(x)
        x = F.relu(x)

        return self._conv_trans_5(x)


class ConvAutoencoderCIFARResidual(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(ConvAutoencoderCIFARResidual, self).__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens, embedding_dim)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        x_comp = self._encoder(x)
        x_recon = self._decoder(x_comp)

        return x_recon, x_comp
