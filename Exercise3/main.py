#!/usr/bin/env python3
# encoding utf-8

import argparse

import torch
import torch.multiprocessing as mp

from Worker import train, computeTargets, computePrediction, saveModelNetwork
from Networks import ValueNetwork
from SharedAdam import SharedAdam

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Asynchronous 1-step Q-learning')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :     

        args = parser.parse_args()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print("Using", device)

        torch.manual_seed(args.seed)
        mp.set_start_method('spawn')

        value_network = ValueNetwork().to(device)
        value_network.share_memory() 

        optimizer = SharedAdam(value_network.parameters(),lr=1e-4)

        counter = mp.Value('i', 0)
        lock = mp.Lock()
        
        I_tar = 2
        I_async = 2

        processes = []
        for idx in range(0, args.num_processes):

                target_value_network = ValueNetwork().to(device)
                target_value_network.share_memory() 
                target_value_network.load_state_dict(value_network.state_dict())

                seed = args.seed + idx
                port = 6000 + 100*idx
                trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter, port, seed, I_tar, I_async)
                p = mp.Process(target=train, args=trainingArgs)
                p.start()
                processes.append(p)
        for p in processes:
                p.join()



