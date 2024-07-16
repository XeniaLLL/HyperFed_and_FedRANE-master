This repository is an official PyTorch implementation of papers:
1. [HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning](https://www.ijcai.org/proceedings/2023/0440.pdf).
IJCAI 2023 (Oral).
2. [FedRANE: Joint Local Relational Augmentation and Global Nash Equilibrium for Federated Learning with Non-IID Data](https://dl.acm.org/doi/abs/10.1145/3581783.3612178)
ACM MM 2023 (Oral)
Thanks to [@PFLlib](https://github.com/TsingZ0/PFLlib) for providing a robust and practical implmentation framework.

# HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning
## Abstract
Federated learning (FL) collaboratively models user data in a decentralized way. However, in the real world, non-identical and independent data distributions (non-IID) among clients hinder the performance of FL due to three issues, i.e., (1) the class
statistics shifting, (2) the insuffcient hierarchical information utilization, and (3) the inconsistency in aggregating clients. To address the above issues, we propose HyperFed which contains three
main modules, i.e., hyperbolic prototype Tammes initialization (HPTI), hyperbolic prototype learning (HPL), and consistent aggregation (CA). Firstly,
HPTI in the server constructs uniformly distributed and fxed class prototypes, and shares them with clients to match class statistics, further guiding consistent feature representation for local clients. Secondly, HPL in each client captures the hierarchical information in local data with the supervision
of shared class prototypes in the hyperbolic model space. Additionally, CA in the server mitigates the impact of the inconsistent deviations from clients
to server. Extensive studies of four datasets prove that HyperFed is effective in enhancing the performance of FL under the non-IID setting.

# FedRANE: Joint Local Relational Augmentation and Global Nash Equilibrium for Federated Learning with Non-IID Data
## Abstract
Federated learning (FL) is a distributed machine learning paradigm that needs collaboration between a server and a series of clients with decentralized data. To make FL effective in real-world applications, existing work devotes to improving the modeling of decentralized non-IID data. In non-IID settings, there are intra-client inconsistency that comes from the imbalanced data modeling, and inter-client inconsistency among heterogeneous client distributions, which not only hinders sufficient representation of the minority data, but also brings discrepant model deviations. However, previous work overlooks to tackle the above two coupling inconsistencies together. In this work, we propose FedRANE, which consists of two main modules, i.e., local relational augmentation (LRA) and global Nash equilibrium (GNE), to resolve intra-and inter-client inconsistency simultaneously. Specifically, in each client, LRA mines the similarity relations among different data samples and enhances the minority sample representations with their neighbors using attentive message passing. In server, GNE reaches an agreement among inconsistent and discrepant model deviations from clients to server, which encourages the global model to update in the direction of global optimum without breaking down the clients' optimization toward their local optimums. We conduct extensive experiments on four benchmark datasets to show the superiority of FedRANE in enhancing the performance of FL with non-IID data.

## Implementation 
This implementation is basically derived from [PFLib](https://github.com/TsingZ0/PFLlib).

###Step 1: generate hyperbolic prototype
```
python system/utils/prototypes.py
```
###Step 2: train model 
HyperFed is hyperbolic prototype based federated learning method.\
MGDA is the consistent updating enhanced hyperbolic prototype based federated learning method.
```
# execute hyperFed-HPL
python main.py -data  Cifar10_new_alpha05 -m resnet -algo HyperbolicFed  -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01

# execute hyperFed-CA
python main.py -data  Cifar10_new_alpha05 -m resnet -algo MGDA  -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01

# execute FedRANE
python main.py -data  Cifar10_new_alpha05 -m resnet -algo FedRANE  -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01

# execute FedRANEAug
python main.py -data  Cifar10_new_alpha05 -m resnet -algo FedRANEAug -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01

```


## Citation
If you find HyperFed useful or inspiring, please consider citing our paper:
```bibtex
@inproceedings{liao2023hyperfed,
  title={HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning},
  author={Liao, Xinting and Liu, Weiming and Chen, Chaochao and Zhou, Pengyang and Zhu, Huabin and Tan, Yanchao and Wang, Jun and Qi, Yue},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)},
  pages={3957--3965},
  year={2023}
}
@inproceedings{liao2023joint,
  title={Joint local relational augmentation and global nash equilibrium for federated learning with non-iid data},
  author={Liao, Xinting and Chen, Chaochao and Liu, Weiming and Zhou, Pengyang and Zhu, Huabin and Shen, Shuheng and Wang, Weiqiang and Hu, Mengling and Tan, Yanchao and Zheng, Xiaolin},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1536--1545},
  year={2023}
}
```

