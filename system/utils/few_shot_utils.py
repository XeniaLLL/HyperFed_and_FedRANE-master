import os
import sys

import numpy as np
import torch
import torch.backends.cudnn
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

zpy77_data_dir = "/home/zpy/code/pfl/dataset"
lxt11_data_dir = "/home/lxt/code/pfl/dataset"
lxt_data_dir = "/home/lxt/code/datasets"

data_PATH = zpy77_data_dir

sys.path.append("../../..")
from dataset.utils.dataset_utils import separate_data


class ImageJitter(object):
    def __init__(self, transform_dict):
        transform_type_dict = dict(
            Brightness=ImageEnhance.Brightness,
            Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness,
            Color=ImageEnhance.Color,
        )
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_tensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_tensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")
        return out


# This is for the CUB dataset, which does not support the ResNet encoder now
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)
class CUB(Dataset):
    def __init__(self, dataset, client_id, is_train):
        IMAGE_PATH = os.path.join(data_PATH, f"{dataset}/data/cub/images")
        if is_train:
            txt_path = os.path.join(data_PATH, f"{dataset}/train/{client_id}.csv")
        else:
            txt_path = os.path.join(data_PATH, f"{dataset}/test/{client_id}.csv")
        lines = [x.strip() for x in open(txt_path, "r").readlines()][1:]
        data = []
        label = []

        for l in lines:
            context = l.split(",")  # e.g. Gadwall_0080_31747.jpg,46
            name = context[0]
            lb = int(context[1])
            path = os.path.join(IMAGE_PATH, name)

            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if is_train:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(84),
                                                 ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])

        else:
            self.transform = transforms.Compose([transforms.Resize(84),
                                                 transforms.CenterCrop(84),
                                                 transforms.ToTensor(),
                                                 normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, label


class MiniImageNet(Dataset):
    def __init__(self, dataset, client_id, is_train):
        IMAGE_PATH = os.path.join(data_PATH, f"{dataset}/data/images")
        if is_train:
            txt_path = os.path.join(data_PATH, f"{dataset}/train/{client_id}.csv")
        else:
            txt_path = os.path.join(data_PATH, f"{dataset}/test/{client_id}.csv")
        lines = [x.strip() for x in open(txt_path, "r").readlines()][1:]
        data = []
        label = []
        for l in lines:
            context = l.split(",")  # e.g. Gadwall_0080_31747.jpg,46
            name = context[0]
            lb = int(context[1])
            path = os.path.join(IMAGE_PATH, name)

            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train == "train":
            self.transform = transforms.Compose([transforms.RandomResizedCrop(84),
                                                 ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize(92),
                                                 transforms.CenterCrop(84),
                                                 transforms.ToTensor(),
                                                 normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, label


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls  # way
        self.n_per = n_per  # shot + query(1 + 1)

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


if __name__ == '__main__':
    import pandas as pd

    """ for bird """
    # num_clients = 10
    # num_classes = 200

    # # change txt to csv
    # text = []
    # fileHandler = open("/home/zpy/code/pfl/dataset/Bird200_test/data/cub/images.txt", "r")
    # while True:
    #     line = fileHandler.readline()
    #     if not line:
    #         break
    #     line = line.strip().split(' ')
    #     result = [line[1].split("/")[1], int(line[1].split(".")[0]) - 1]
    #     text.append(result)
    # fileHandler.close()
    # df = pd.DataFrame(text)
    # df.to_csv("/home/zpy/code/pfl/dataset/Bird200_test/data/cub/images.csv", index=False)

    # # split data to clients
    # import sys
    #
    # sys.path.append("../../..")
    # from dataset.utils.dataset_utils import separate_data
    #
    # data = pd.read_csv(r"/home/zpy/code/pfl/dataset/Bird200_test/data/cub/images.csv")
    # input_data = (np.array(data["filename"]), np.array(data["label"]))
    # X, y, statistics = separate_data(input_data, num_clients, num_classes, balance=True)
    # for i in range(num_clients):
    #     save_data = {"filename": X[i], "label": y[i]}
    #     df = pd.DataFrame(data=save_data)
    #     file_name = f"/home/zpy/code/pfl/dataset/Bird200_test/data/cub/images{i}.csv"
    #     df.to_csv(file_name, index=False)
    #
    # # split train and test data
    # for i in range(num_clients):
    #     data = pd.read_csv(f"/home/zpy/code/pfl/dataset/Bird200_test/data/cub/images{i}.csv")
    #     # X_train, X_test, y_train, y_test = train_test_split(np.array(data["filename"]), np.array(data["label"]),
    #     #                                                     train_size=0.75, shuffle=True)
    #     # train_data = pd.DataFrame(data={"filename": X_train, "label": y_train})
    #     # test_data = pd.DataFrame(data={"filename": X_test, "label": y_test})
    #     # train_data.to_csv(f"/home/zpy/code/pfl/dataset/Bird200_test/train/{i}.csv", index=False)
    #     # test_data.to_csv(f"/home/zpy/code/pfl/dataset/Bird200_test/test/{i}.csv", index=False)
    #
    #     train_data = pd.DataFrame()
    #     test_data = pd.DataFrame()
    #     for class_id in range(num_classes):
    #         test_data = pd.concat([test_data, data[data["label"] == class_id][:2]])
    #         train_data = pd.concat([train_data, data[data["label"] == class_id][2:]])
    #     train_data.to_csv(f"/home/zpy/code/pfl/dataset/Bird200_test/train/{i}.csv", index=False)
    #     test_data.to_csv(f"/home/zpy/code/pfl/dataset/Bird200_test/test/{i}.csv", index=False)

    """ for MiniImagenet """
    num_clients = 10
    num_train_classes = 80
    num_test_classes = 20
    dataset_name = "MiniImagenet"

    # split data to clients
    # train data
    train_data = pd.read_csv(f"/home/zpy/code/pfl/dataset/{dataset_name}/train.csv")
    val_data = pd.read_csv(f"/home/zpy/code/pfl/dataset/{dataset_name}/val.csv")
    train_data = pd.concat([train_data, val_data], ignore_index=True)
    label_list = train_data["label"].unique()
    label_dict = dict()
    for i, label in enumerate(label_list):
        label_dict[label] = i
    train_data.replace(label_dict, inplace=True)
    input_train = (np.array(train_data["filename"]), np.array(train_data["label"]))
    X, y, statistics = separate_data(input_train, num_clients, num_train_classes, balance=True)
    for i in range(num_clients):
        save_data = {"filename": X[i], "label": y[i]}
        df = pd.DataFrame(data=save_data)
        file_name = f"/home/zpy/code/pfl/dataset/{dataset_name}/train/{i}.csv"
        df.to_csv(file_name, index=False)

    # test data
    test_data = pd.read_csv(f"/home/zpy/code/pfl/dataset/{dataset_name}/test.csv")
    label_list = test_data["label"].unique()
    label_dict = dict()
    for i, label in enumerate(label_list):
        label_dict[label] = i
    test_data.replace(label_dict, inplace=True)
    input_test = (np.array(test_data["filename"]), np.array(test_data["label"]))
    X, y, statistics = separate_data(input_test, num_clients, num_test_classes, balance=True)
    for i in range(num_clients):
        save_data = {"filename": X[i], "label": y[i]}
        df = pd.DataFrame(data=save_data)
        file_name = f"/home/zpy/code/pfl/dataset/{dataset_name}/test/{i}.csv"
        df.to_csv(file_name, index=False)
