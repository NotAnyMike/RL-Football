from pdb import set_trace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train(idx, args, value_network, target_value_network, optimizer, lock, counter, port, seed, I_tar, I_async):

        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
        hfoEnv.connectToServer()

        episodeNumber = 0
        gamma = 0.99
        epsilon = 0.8
        t = 0
        loss_func = nn.MSELoss()
        state1 = hfoEnv.reset()
        for epoch in range(1, args.epochs + 1):
                #train_epoch(epoch, args, model, device, train_loader, optimizer)
                if np.random.random() >= epsilon:
                    action = random.randint(0,3)
                else:
                    qs = [computePrediction(state1,a,value_network) for a in range(4)]
                    action = np.argmax(qs)

                a1 = hfoEnv.possibleActions[action]
                a1_num = hfoEnv.possibleActions.index(a1)

                state2, reward, done, status, info = hfoEnv.step(a1)
                #print(state2, reward, done, status, info)

                if done:
                        episodeNumber += 1

                y = computeTargets(reward, state2, gamma, done, target_value_network)

                prediction = computePrediction(state1,a1_num, value_network)
                loss = loss_func(prediction, y)
                loss.backward()

                state1 = state2
                t += 1
                with lock:
                        counter.value += counter.value + 1

                if t % I_async == 0 or done:
                        # Async update of value_network using gradients

                        with lock:
                                # Add grads to value_network
                                for param, shared_param in zip(
                                            value_network.parameters(), 
                                            target_value_network.parameters()):
                                        shared_param._grad = param.grad
                                        #value_network._grad = target_value_network.grad
                                # Take a step
                                optimizer.step()
                                # Clean gradients
                                optimizer.zero_grad()
                        target_value_network.zero_grad()

                if counter.value % I_tar == 0:
                        # Update target network
                        target_value_network.zero_grad()
                        target_value_network.load_state_dict(value_network.state_dict())

        #torch.manual_seed(args.seed + rank)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
        if done:
                y = torch.autograd.Variable(torch.cuda.FloatTensor([reward]))
        else:
                qs = [computePrediction(nextObservation,a,targetNetwork) for a in range(4)]
                q_max = np.max(qs)
                y = reward + discountFactor*q_max

        return y

def computePrediction(state, action, valueNetwork, possible_actions=None):
        '''
        action should be a number
        '''
        if possible_actions == None:
                # to allow changing to different action space
                possible_actions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']
        if type(action) == str: 
                action = possible_actions.index(action)
        inputs = np.concatenate([state,[action]])
        inputs = torch.from_numpy(inputs)
        return valueNetwork(inputs.to('cuda'))
        
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

