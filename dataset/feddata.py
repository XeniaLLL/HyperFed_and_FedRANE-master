from typing import Optional, List, Tuple, Dict, TypedDict
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(1998)


class ClientData(TypedDict):
    train_xs: np.ndarray
    train_ys: np.ndarray
    test_xs: np.ndarray
    test_ys: np.ndarray


class FedData():
    '''
    Different split ways to support FL
    '''

    def __init__(self,
                 dataset: str = "mnist",
                 test_ratio: float = 0.1,
                 split: Optional[str] = None,
                 n_clients: Optional[int] = None,
                 nc_per_client: Optional[int] = None,
                 n_client_perc=None,
                 dir_alpha: float = 1.0,
                 n_max_sam: Optional[int] = None):
        '''

        Args:
            dataset: candidates in [fmnist, cifar10]
            test_ratio:
            split: split by, candidates in [label, user, None],
                if split by "user", split each user to a client;
                if split by "label", split to n_clients w/ samples from several class
            n_clients: int, None; consistent with split method
            nc_per_client: # of classes per client <- only for split by label
            n_client_perc: # of clients per class <- only for split by label and dataset== sa #question what is sa
            dir_alpha: parameter alpha for dirichlet distribution
            n_max_sam: max # of samples per client -> for low-resource learning
        '''
        self.dataset = dataset
        self.test_ratio = test_ratio
        self.split = split
        self.n_clients = n_clients
        self.nc_per_client = nc_per_client
        self.n_client_perc = n_client_perc
        self.dir_alpha = dir_alpha
        self.n_max_sam = n_max_sam

        self.label_dsets = ['fmnist', "cifar10", "cifar100"]  # note and so on

        if dataset in self.label_dsets:
            assert self.split in ['label', 'dirichlet']
            assert (n_clients is not None), f" {dataset} needs pre-defined n_clients"
            if self.split == "label":
                if dataset == "sa":
                    assert (n_client_perc is not None), f"{dataset} needs pre-defined n_client_perc"
                else:
                    assert (nc_per_client is not None), f"{dataset} needs pre-defined nc_per_client"

    def split_by_dirichlet(self, xs, ys):
        '''
        split data into N clients w/ distribution w/ Diri(alpha)
        Args:
            xs: (N, ...)
            ys: (N, )

        Returns:
            dict like: client:{train_xs, train_ys, test_xs, test_ys}

        '''
        # unique classes
        n_classes = len(np.unique(ys))
        class_cnts = np.array([np.sum(ys == c) for c in range(n_classes)])
        class_indexs = {c: np.argwhere(ys == c).reshape(-1) for c in range(n_classes)}

        # (n_clients, n_classes) note: 构造采样的分布-> 服从dirichlet
        dists = np.random.dirichlet(
            alpha=[self.dir_alpha] * n_classes,
            size=self.n_clients
        )
        dists = dists / dists.sum(axis=0)

        # (n_clients, n_classes)
        cnts = (dists * class_cnts.reshape((1, -1)))
        cnts = np.round(cnts).astype(np.int32)  # note 取整

        cnts = np.cumsum(cnts, axis=0)
        cnts = np.concatenate([np.zero((1, n_classes)).astype(np.int32),
                               cnts], axis=0)

        # split data by Dists
        clients_data = {}
        for n in range(self.n_clients):
            client_xs = []
            client_ys = []
            for c in range(n_classes):
                cinds = class_indexs[c]
                bi, ei = cnts[n][c], cnts[n + 1][c]  # begin and end index
                c_xs = xs[cinds[bi:ei]]
                c_ys = ys[cinds[bi:ei]]

                client_xs.append(c_xs)
                client_ys.append(c_ys)
                if n == self.n_clients - 1:
                    print(c, len(cinds), bi, ei)
            client_xs = np.concatenate(client_xs, axis=0)
            client_ys = np.concatenate(client_ys, axis=0)
            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]
            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[n] = {
                'train_xs': client_xs[n_test: n_end],
                'train_ys': client_xs[n_test: n_end],
                'test_xs': client_xs[: n_test],
                'test_ys': client_ys[: n_test]
            }
        return clients_data

    def split_by_label(self, xs, ys):
        '''
        split data into N client, each clients has k (k \leq # of classes) classes
        Args:
            xs: (N, ...)
            ys: (N,)

        Returns:
            dict like: client:{train_xs, train_ys, test_xs, test_ys}

        '''
        # unique classes
        uni_classes = sorted(np.unique(ys))
        print("unique classes:", uni_classes)
        print("number of unique classes:", len(uni_classes))
        seq_classes = []
        for _ in range(self.n_clients):
            np.random.shuffle(uni_classes)
            seq_classes.extend(list(uni_classes))
        print("sequence classes:", seq_classes)
        print("number of sequence classes:", len(seq_classes))

        # each class at least assigned to a client
        assert (self.nc_per_client * self.n_clients >= len(uni_classes)), " Each class at least assigned to a client"
        # assign classes to each client
        client_classes = {}
        for k, client in enumerate(range(self.n_clients)):  # NOTE 分箱问题
            client_classes[client] = seq_classes[
                                     k * self.nc_per_client: (k + 1) * self.nc_per_client
                                     ]
        print("client classes: ", client_classes)

        # for a class, how many clients have it note 统计
        classes = []
        for client in client_classes.keys():
            classes.extend(client_classes[client])
        print("classes: ", classes)
        classes_cnt = dict(Counter(classes))
        print("classes cnt:", classes_cnt)

        n_samples = xs.shape[0]
        print('number of samples:', n_samples)
        inds = np.random.permutation(n_samples)
        xs = xs[inds]
        ys = ys[inds]

        # assign classes to each client
        clients_data = {}
        for client in client_classes.keys():
            clients_data[client] = {
                'xs': [],
                'ys': []
            }

        # split data by classes
        for c in uni_classes:
            cinds = np.argwhere(ys == c).reshape(-1)
            c_xs = xs[cinds]
            c_ys = ys[cinds]

            # assign class data uniformly to each client
            t = 0
            for client, client_cs in client_classes.items():
                if c in client_cs:
                    n = client_cs.count(c)
                    ind1 = t * int(len(c_xs) / classes_cnt[c])  # note 定位class 所在的位置
                    ind2 = (t + n) * int(len(c_xs) / classes_cnt[c])
                    print(f"ind1: {ind1}, ind2: {ind2}")
                    clients_data[client]["xs"].append(c_xs[ind1:ind2])
                    clients_data[client]["ys"].append(c_ys[ind1:ind2])
                    t += n
            assert (t == classes_cnt[c]), f"Error, t != classes_cnt[c] for c =={c}"

        # shuffle and limit maximum number
        for client, values in clients_data.items():
            client_xs = np.concatenate(values["xs"], axis=0)
            client_ys = np.concatenate(values["ys"], axis=0)

            print(f"shape for client_xs: {client_xs.shape}")
            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)  # note 传入的是test_ratio 取前面的为测试集
            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            print(f"n_test: {n_test}, n_end: {n_end}")

            clients_data[client] = {
                'train_xs': client_xs[n_test: n_end],
                'train_ys': client_xs[n_test: n_end],
                'test_xs': client_xs[: n_test],
                'test_ys': client_ys[: n_test]
            }
        return clients_data

    def split_uniformly(self, xs: np.ndarray, ys: np.ndarray) -> Dict[int, ClientData]:
        """
        Split data and labels into clients uniformly.

        Please set `n_clients` before using this method.

        Args:
            xs: shape(N, ...)
            ys: shape(N, )

        Returns:
            Dictionary of client_id => client_data.
        """
        if self.n_clients is None:
            raise TypeError('split_uniformly requires n_clients')
        clients = _split_uniformly(xs, ys, self.n_clients)
        clients_train_test = _client_train_test_split(clients, self.test_ratio)
        return {
            i: {
                'train_xs': train_data,
                'test_xs': test_data,
                'train_ys': train_labels,
                'test_ys': labels
            } for i, (train_data, test_data, train_labels, labels) in enumerate(clients_train_test)
        }

    def split_dirichlet_quantity(self, xs: np.ndarray, ys: np.ndarray, min_size: int = 5) -> Dict[int, ClientData]:
        """
        Split data and labels into clients. The number of samples per client
        is generated by Dirichlet distribution.

        Please set `n_clients` before using this method.

        Args:
            xs: shape(N, ...)
            ys: shape(N, )
            min_size: The number of samples per client should be greater than
                or equal to `min_size`.

        Returns:
            Dictionary of client_id => client_data.
        """
        if self.n_clients is None:
            raise TypeError('split_dirichlet_quantity requires n_clients')
        clients = _split_dirichlet_quantity(xs, ys, self.n_clients, self.dir_alpha, min_size=min_size)
        clients_train_test = _client_train_test_split(clients, self.test_ratio)
        return {
            i: {
                'train_xs': train_data,
                'test_xs': test_data,
                'train_ys': train_labels,
                'test_ys': labels
            } for i, (train_data, test_data, train_labels, labels) in enumerate(clients_train_test)
        }

    def construct_datasets(self, clients_data, Dataset, glo_test_xs=None, glo_test_ys=None):
        '''

        Args:
            clients_data:
            Dataset:
            glo_test_xs:
            glo_test_ys:

        Returns:
            client: (train_set, test_set)

        '''
        csets = {}
        if glo_test_xs is None or glo_test_ys is None:
            glo_test = False
        else:
            glo_test = True

        if glo_test is False:
            glo_test_xs = []
            glo_test_ys = []

        for client, cdata in clients_data.items():
            train_set = Dataset(
                cdata['train_xs'], cdata['train_ys'], is_train=True
            )
            test_set = Dataset(
                cdata['test_xs'], cdata['test_ys'], is_train=False
            )
            csets[client] = (train_set, test_set)
            if glo_test is False:
                glo_test_xs.append(cdata["test_xs"])
                glo_test_ys.append(cdata["test_ys"])

        if glo_test is False:
            glo_test_xs = np.concatenate(glo_test_xs, aixs=0)
            glo_test_ys = np.concatenate(glo_test_ys, aixs=0)

        gset = Dataset(glo_test_xs, glo_test_ys, is_train=False)
        return csets, gset

    def split_by_label_noniid(self, xs, ys):
        if self.split == "label":
            clients_data = self.split_by_label(xs, ys)
        elif self.split == "dirichlet":
            clients_data = self.split_by_dirichlet(xs, ys)
        elif self.split == 'uniform':
            clients_data = self.split_uniformly(xs, ys)
        elif self.split == 'dirichlet_quantity':
            clients_data = self.split_dirichlet_quantity(xs, ys)
        else:
            raise ValueError(f"No such split: {self.split} is supported.")
        return clients_data

    def construct(self):
        '''load raw data'''
        if self.dataset=="fmnist":
            pass
            # step 1: load dataset by numpy.ndarray form
            # step 2: self.split_by_label_noniid()
            # step 3: self.construct_datasets()

    def print_info(self, csets, gset, max_cnt=10):
        # note uncheck todo
        """ print information
        """
        print("#" * 50)
        cnt = 0
        print("Dataset:{}".format(self.dataset))
        print("N clients:{}".format(len(csets)))

        for client, (cset1, cset2) in csets.items():
            print("Information of Client {}:".format(client))
            print(
                "Local Train Set: ", cset1.xs.shape,
                cset1.xs.max(), cset1.xs.min(), Counter(cset1.ys)
            )
            print(
                "Local Test Set: ", cset2.xs.shape,
                cset2.xs.max(), cset2.xs.min(), Counter(cset2.ys)
            )

            cnts = [n for _, n in Counter(cset1.ys).most_common()]
            probs = np.array([n / sum(cnts) for n in cnts])
            ent = -1.0 * (probs * np.log(probs + 1e-8)).sum()
            print("Class Distribution, Min:{}, Max:{}, Ent:{}".format(
                np.min(probs), np.max(probs), ent
            ))

            if cnt >= max_cnt:
                break
            cnt += 1

        print(
            "Global Test Set: ", gset.xs.shape,
            gset.xs.max(), gset.xs.min(), Counter(gset.ys)
        )
        print("#" * 50)


def _split_uniformly(
    data: np.ndarray,
    labels: np.ndarray,
    n_clients: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data and labels into clients uniformly.

    Args:
        data: shape(N, ...)
        labels: shape(N, )

    Returns:
        List of (data, labels).
    """
    data, labels = _shuffle_together(data, labels)
    return list(zip(np.array_split(data, n_clients), np.array_split(labels, n_clients)))


def _split_dirichlet_quantity(
    data: np.ndarray,
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    min_size: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data and labels into clients. The number of samples per client
    is generated by Dirichlet distribution.

    Args:
        data: shape(N, ...)
        labels: shape(N, )
        min_size: The number of samples per client should be greater than
            or equal to `min_size`.

    Returns:
        List of (data, labels).
    """
    n = len(data)
    assert n_clients * min_size <= n
    data, labels = _shuffle_together(data, labels)
    while True:
        proportion = np.random.dirichlet(np.repeat(alpha, n_clients))  # shape(n_clients, )
        sample_nums = (n * proportion).astype(np.int64)
        if sample_nums.min() >= min_size:
            break
    split_points = np.cumsum(sample_nums)[:-1]
    return list(zip(np.array_split(data, split_points), np.array_split(labels, split_points)))


def _shuffle_together(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Shuffle multiple arrays together. Multi-dimensional arrays will be
    shuffled along the first axis.
    """
    n = len(arrays[0])
    if any(len(array) != n for array in arrays):
        raise ValueError('All arrays must be of the same length.')
    indices = np.random.permutation(n)
    return tuple(array[indices] for array in arrays)


def _client_train_test_split(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    test_ratio: float,
    filter_small_corpus_size: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Args:
        clients: List of (data, labels) of each client.
        test_ratio: ratio of test set.

    Returns:
        List of (train_data, test_data, train_labels, test_labels).
    """
    return [
        tuple(train_test_split(data, labels, test_size=test_ratio))
        for data, labels in clients if filter_small_corpus_size is None or len(data) >= filter_small_corpus_size
    ]
