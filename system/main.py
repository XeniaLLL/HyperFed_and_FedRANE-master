#!/usr/bin/env python
import copy
import math
import os
os.environ["CUDA_VISIBLE_DEVICEdS"] ="1"# args.device_id
# os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
import torch
import argparse
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision
from torchvision import models
import wandb

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverhyperbolic import HyperbolicFed
from flcore.servers.servermgda import MGDA

from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
from utils.mem_utils import MemReporter

# from utils.plot import plot_result

warnings.simplefilter("ignore")
torch.manual_seed(0)
torch.cuda.set_device('cuda:1')
# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32

def run():
    torch.manual_seed(0)
    args = parser.parse_args()
    if not args.debug:
        wandb.init()
        args.batch_size = wandb.config.batch_size
        args.local_steps = wandb.config.local_steps
        args.local_learning_rate = wandb.config.local_learning_rate
        # args.margin_triplet = wandb.config.margin_triplet
        # args.tau = wandb.config.tau
        # args.mu = wandb.config.mu
        args.fine_tuning_steps = wandb.config.fine_tuning_steps
        args.mult_slope = wandb.config.mult_slope
        args.HyperbolicFed_dim = wandb.config.HyperbolicFed_dim
        # args.global_rounds = int(900 / args.local_steps)
        args.clip_r = wandb.config.clip_r
        args.multi_task_method = wandb.config.multi_task_method
        wandb.run.name = f"simple-test"


    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":
            if args.dataset.lower() == "mnist" or args.dataset.lower() == "fmnist":
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":
            if "mnist" in args.dataset.lower():
                # if args.dataset.lower() == "mnist" or args.dataset.lower() == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                # args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif args.dataset[:13].lower() == "tiny-imagenet" or args.dataset[:8].lower() == "imagenet":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        elif model_str == "dnn":  # non-convex
            # if args.dataset.lower() == "mnist" or args.dataset.lower() == "fmnist":
            if "mnist" in args.dataset.lower():
                args.model = DNN(1 * 28 * 28, mid_dim=20, num_classes=args.num_classes).to(
                    args.device)
            elif args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
                args.model = DNN(3 * 32 * 32, mid_dim=20, num_classes=args.num_classes).to(
                    args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "ConvNet":
            if "mnist" in args.dataset.lower():
                args.model = CNNModel(in_features=1, num_classes=args.num_classes, dim=1568).to(args.device)
            elif "cifar" in args.dataset.lower():
                # args.model = ConvNet(in_features=3, num_classes=args.num_classes).to(args.device)
                args.model = CNNModel(in_features=3, num_classes=args.num_classes, dim=2048).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False).to(args.device)
            # args.model = torchvision.models.resnet18(pretrained=args.use_pretrain_resnet).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)
            if "mnist" in args.dataset.lower():
                # if args.dataset.lower() == "emnist_alpha05" or args.dataset.lower() == "emnist_letters_alpha05":
                args.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(
                    args.device)

        elif model_str == "resnet4":
            args.model = resnet4(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet6":
            args.model = resnet6(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet8":
            args.model = resnet8(num_classes=args.num_classes).to(args.device)


        # elif model_str == "resnet20":
        #     args.model = resnet20(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet50":
            # args.model = torchvision.models.resnet50(num_classes=args.num_classes).to(args.device)
            args.model = torchvision.models.resnet50(pretrained=True).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)


        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            # args.model.fc = nn.Linear(in_features=args.model.fc.in_features, out_features=128).to(args.device)
            # args.predictor = nn.Linear(in_features=128, out_features=args.num_classes).to(args.device)
            # args.model = LocalModel(args.model, args.predictor)
            server = FedAvg(args, i)

        elif args.algorithm == "MGDA":

            args.model.fc = nn.Linear(in_features=args.model.fc.in_features, out_features=args.HyperbolicFed_dim).to(

                args.device)

            args.predictor = nn.Linear(in_features=args.HyperbolicFed_dim, out_features=args.num_classes).to(

                args.device)

            args.predictor.weight.data = torch.from_numpy(

                np.load(args.hyperbolic_proto_dir.format(args.HyperbolicFed_dim))).float().to(

                args.device) * args.mult_slope

            args.predictor.bias.data = torch.zeros_like(args.predictor.bias.data).to(args.device)

            args.predictor.requires_grad_(False)

            args.model = LocalModel(args.model, args.predictor)

            server = MGDA(args, i)

        elif args.algorithm == "HyperbolicFed":
            args.model.fc = nn.Linear(in_features=args.model.fc.in_features, out_features=args.HyperbolicFed_dim).to(
                args.device)
            args.predictor = nn.Linear(in_features=args.HyperbolicFed_dim, out_features=args.num_classes).to(
                args.device)

            # todo tammes opt every moment
            classpolars = torch.from_numpy(
                np.load(args.hyperbolic_proto_dir.format(args.HyperbolicFed_dim, args.num_classes))).float().to(
                args.device) * args.mult_slope
            # calculate radius of ball
            radius = 1. / math.sqrt(args.curvature)
            args.classpolars = classpolars * radius * args.mult_slope
            args.predictor.weight.data = args.classpolars
            args.predictor.bias.data = torch.zeros_like(args.predictor.bias.data).to(args.device)  # careful
            args.predictor.requires_grad_(False)  # careful 在这里先说明不允许
            # args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = HyperbolicFed(args, i)

        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    # reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="dnn")
    parser.add_argument('-p', "--predictor", type=str, default="dnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_steps", type=int, default=10)
    parser.add_argument('-algo', "--algorithm", type=str, default="ablationG")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    parser.add_argument('-spe', "--save_per_epoch", type=int, default=10,
                        help="interval for saving the checkpoint")
    parser.add_argument('-cn', "--checkpoint_name", type=str, default='HyperbolicFed')
    parser.add_argument('-dbg', "--debug", type=bool, default=False)
    parser.add_argument("-d", "--dimension", type=int, default=1)  # representation dimension

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threshold", type=float, default=10000,
                        help="The threshold for dropping slow clients")
    parser.add_argument('-suf', "--suffix", type=str, default="", help="suffix of results filename")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0.0005,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to calculate theta approximately using K steps")

    # HyperbolicFed & MGDA
    parser.add_argument('-hb_p_dir', "--hyperbolic_proto_dir", type=str,
                        # default="/home/lxt/code/pfl/prototypes/hyperbolic-prototypes-{}-{}.npy")
                        default="/home/zpy/code/pfl/prototypes/hyperbolic-prototypes-{}-{}.npy")
    parser.add_argument('-curv', "--curvature", type=float, default=1.)
    parser.add_argument('-hb_dim', "--HyperbolicFed_dim", type=int, default=20)
    parser.add_argument('-mult', "--mult_slope", type=float, default=0.9)
    parser.add_argument('-visual', "--visualize", type=bool, default=False)

    # HyperbolicFed for CL
    parser.add_argument('-mrgin_tri', "--margin_triplet", type=float, default=1.)

    parser.add_argument('-testpm', "--test_pm", type=bool, default=False,
                        help="Use the distribution probability to test")


    # few_shot learning
    parser.add_argument("-shot", "--shot", type=int, default=1)
    parser.add_argument("-way", "--way", type=int, default=5)
    parser.add_argument("-query", "--query", type=int, default=1)


    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threshold: {}".format(args.time_threshold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    if args.debug:
        run()

    else:
        sweep_configuration = {
            'method': 'grid',
            # 'metric': {'goal': 'maximize',
            #            'name': 'test_accuracy'},
            'parameters': {
                'HyperbolicFed_dim': {
                    'values': [20]
                },
            }
        }

        sweep_id = wandb.sweep(sweep_configuration, project="YOUR PROJECT NAME")
        wandb.agent(sweep_id, function=run)

        wandb.finish()
