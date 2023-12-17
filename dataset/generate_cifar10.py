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
def generate_cifar10(
    dir_path: str,
    num_clients: int,
    num_classes: int,
    niid: bool,
    balance: bool,
    partition: str
):
    """Allocate data to users"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    dataset_image, dataset_label = cifar10_ndarray(os.path.join(dir_path, 'rawdata'))

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
    dir_path = "/home/lxt/code/datasets/cifar10/"
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)
# endregion


def cifar10_ndarray(root: str):
    return vision_dataset_ndarray(
        root=root,
        dataset_type=torchvision.datasets.CIFAR10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )


args = parse_cmd_args()


def main():
    """
    An example of running this Python script:

    ```shell
    python generate_cifar10.py \
        --dir_path=path/to/dataset \
        --n_clients=20 \
        --strategy=dirichlet \
        --test_ratio=0.25 \
        --alpha=0.1 \
        --min_size=4
    ```
    """
    random.seed(1)
    np.random.seed(1)
    data, labels = cifar10_ndarray(os.path.join(args.dir_path, 'rawdata'))
    main_template(args, data=data, labels=labels)


if __name__ == "__main__":
    main()
