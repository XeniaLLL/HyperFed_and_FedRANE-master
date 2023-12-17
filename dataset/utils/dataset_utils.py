import os
import os.path
import gc
from typing import Optional, Literal, List, Tuple, TypedDict

import ujson
import numpy as np
from sklearn.model_selection import train_test_split

from ._split_strategies import (
    split_by_label,
    split_dirichlet_label,
    split_dirichlet_quantity,
    split_uniformly
)


batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = batch_size / (1-train_size) # least samples for each client
alpha = 0.1 # for Dirichlet distribution

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(
    data: Tuple[np.ndarray, np.ndarray],
    num_clients: int,
    num_classes: int,
    niid: bool = False,
    balance: bool = False,
    partition = None,
    class_per_client: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic: List[List[Tuple[int, int]]] = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


class DataDict(TypedDict):
    x: np.ndarray
    y: np.ndarray


def split_data(X: List[np.ndarray], y: List[np.ndarray]) -> Tuple[List[DataDict], List[DataDict]]:
    # Split dataset
    train_data: List[DataDict] = []
    test_data: List[DataDict] = []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data



def __global_alpha() -> float:
    return alpha

def save_file(
    config_path: str,
    train_path: str,
    test_path: str,
    train_data: List[DataDict],
    test_data: List[DataDict],
    num_clients: int, 
    num_classes: int,
    statistic: List[List[Tuple[int, int]]],
    niid: bool = False,
    balance: bool = True,
    partition: Optional[str] = None,
    alpha: Optional[float] = None):
    if alpha is None:
        alpha = __global_alpha()
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def split_with_strategy(
    data: np.ndarray,
    labels: np.ndarray,
    n_clients: int,
    strategy: Literal['label', 'dirichlet', 'uniform', 'dirichlet_quantity'],
    *,
    alpha: Optional[float] = None,
    min_size: Optional[int] = None,
    n_class_per_client: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Randomly split data and labels into clients with IID or non-IID splitting strategy.

    Args:
        data: shape(N, ...)
        labels: shape(N, )
        n_clients: number of clients
        strategy:
            - 'uniform': use `split_uniformly`
            - 'label': use `split_by_label`
            - 'dirichlet': use `split_dirichlet_label`
            - 'dirichlet_quantity': use `split_dirichlet_quantity`
        alpha: parameter of 'dirichlet' and 'dirichlet_quantity' splitting.
        min_size: parameter of 'dirichlet' and 'dirichlet_quantity' splitting, default value is 1.
        n_class_per_client: parameter of 'label' splitting.

    Returns:
        List of (data, labels).
    """
    if strategy == 'label':
        if n_class_per_client is None:
            raise ValueError("split by label requires parameter 'n_class_per_client'")
        return split_by_label(data, labels, n_clients, n_class_per_client)
    elif strategy == 'dirichlet':
        if alpha is None:
            raise ValueError("split by dirichlet label requires parameter 'alpha'")
        if min_size is None:
            min_size = 1
        return split_dirichlet_label(data, labels, n_clients, alpha)
    elif strategy == 'uniform':
        return split_uniformly(data, labels, n_clients)
    elif strategy == 'dirichlet_quantity':
        if alpha is None:
            raise ValueError("split by dirichlet quantity requires parameter 'alpha'")
        if min_size is None:
            min_size = 1
        return split_dirichlet_quantity(data, labels, n_clients, alpha=alpha, min_size=min_size)
    else:
        raise ValueError(f'unsupported split strategy: {strategy}')


def train_test_split_with_strategy(
    data: np.ndarray,
    labels: np.ndarray,
    n_clients: int,
    strategy: Literal['label', 'dirichlet', 'uniform', 'dirichlet_quantity'],
    test_ratio: float,
    *,
    alpha: Optional[float] = None,
    min_size: int = 1,
    n_class_per_client: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Combine `split_with_strategy` and `train_test_split`. For detail information,
    see `split_with_strategy`.

    Returns:
        List of (train_data, test_data, train_labels, test_labels).
    """
    clients = split_with_strategy(
        data,
        labels,
        n_clients,
        strategy,
        alpha=alpha,
        min_size=min_size,
        n_class_per_client=n_class_per_client
    )
    return [
        tuple(train_test_split(data, labels, test_size=test_ratio))
        for data, labels in clients
    ]
