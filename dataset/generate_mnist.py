import os
import os.path
import sys
import random

import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms

from utils.dataset_utils import check, separate_data, split_data, save_file
from generate_templates import parse_cmd_args, main_template, vision_dataset_ndarray


# region: legacy code
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
    """Allocate data to users"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    dataset_image, dataset_label = mnist_ndarray(os.path.join(dir_path, 'rawdata'))

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


def main_legacy():
    random.seed(1)
    np.random.seed(1)
    num_clients = 20
    num_classes = 10
    dir_path = "/home/lxt/code/datasets/MNIST/" # careful absolute path for mnist data
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition)
#endregion


def mnist_ndarray(root: str):
    return vision_dataset_ndarray(
        root=root,
        dataset_type=torchvision.datasets.MNIST,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )


args = parse_cmd_args()


def main():
    random.seed(1)
    np.random.seed(1)
    data, labels = mnist_ndarray(os.path.join(args.dir_path, 'rawdata'))
    main_template(args, data=data, labels=labels)


if __name__ == "__main__":
    main()
