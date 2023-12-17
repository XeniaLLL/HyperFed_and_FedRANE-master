# HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning
This repository is an official PyTorch implementation of paper:
[HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning](https://www.ijcai.org/proceedings/2023/0440.pdf).
IJCAI 2023 (Oral).

Thanks to [@PFLlib](https://github.com/TsingZ0/PFLlib) for providing a robust and practical implmentation framework.


## Abstract
Federated learning (FL) collaboratively models user data in a decentralized way. However, in the real world, non-identical and independent data distributions (non-IID) among clients hinder the performance of FL due to three issues, i.e., (1) the class
statistics shifting, (2) the insuffcient hierarchical information utilization, and (3) the inconsistency in aggregating clients. To address the above issues, we propose HyperFed which contains three
main modules, i.e., hyperbolic prototype Tammes initialization (HPTI), hyperbolic prototype learning (HPL), and consistent aggregation (CA). Firstly,
HPTI in the server constructs uniformly distributed and fxed class prototypes, and shares them with clients to match class statistics, further guiding consistent feature representation for local clients. Secondly, HPL in each client captures the hierarchical information in local data with the supervision
of shared class prototypes in the hyperbolic model space. Additionally, CA in the server mitigates the impact of the inconsistent deviations from clients
to server. Extensive studies of four datasets prove that HyperFed is effective in enhancing the performance of FL under the non-IID setting.

## Implementation 
This implementation is basically derived from [PFLib](https://github.com/TsingZ0/PFLlib).
###Step 1: generate hyperbolic prototype

###Step 2: train model 
HyperFed is hyperbolic prototype based federated learning method.\
MGDA is the consistent updating enhanced hyperbolic prototype based federated learning method.
```
# execute hyperFed-HPL
python main.py -data  Cifar10_new_alpha05 -m resnet -algo HyperbolicFed  -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01

# execute hyperFed-CA
python main.py -data  Cifar10_new_alpha05 -m resnet -algo MGDA  -did 1 -nc 20 -lbs 128 -gr 100 --num_classes 10 --debug False -lr 0.01
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
```

