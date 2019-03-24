from pdb import set_trace
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tensorboard_logger import Logger
from hfo import GOAL

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train(idx, args, value_network, target_value_network, optimizer, lock, counter, 
                port, seed, I_tar, I_async, name=None):

        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed, headless=True)
        hfoEnv.connectToServer()

        if name == None:
                name = str(time())
        logger = Logger("tb/" + name, flush_secs=5)

        if args.eval:
                num_episodes = (args.eval_episodes)
        else:
                target_value_network.train()
                num_episodes = (args.episodes, args.eval_episodes)

        gamma = 0.99
        windows = [10, 500]
        goal_buffer = [0]*max(windows)

        max_epsilon = 0.99
        min_epsilon = 0.1
        total = 200
        epsilon_fn = lambda current : max_epsilon - current*(max_epsilon - min_epsilon)/total if current < total else min_epsilon

        max_lr = 0.9
        min_lr = 0.1
        lr_fn = lambda current: max_lr - current*(max_lr - min_lr)/total if current < total else min_lr

        for episodes in num_episodes:

                loss_func = nn.MSELoss()
                t = 0
                episodeNumber = 0
                episodeReward = 0
                episodeSteps  = 0
                state1 = hfoEnv.reset()
                
                if args.eval:
                        print("##################################################")
                        print("#################### Evaluation ##################")
                        print("##################################################")

                while episodeNumber <= episodes:

                        if t % 1e6 == 0:
                            saveModelNetwork(value_network,"trained_models/params"+str(int(t // 1e6)))


                        #train_epoch(epoch, args, model, device, train_loader, optimizer)
                        epsilon = epsilon_fn(episodeNumber)
                        lr = lr_fn(episodeNumber)

                        if args.eval or np.random.random() >= epsilon:
                                qs = value_network(state1)
                                action = torch.argmax(qs)
                        else:
                                action = random.randint(0,len(hfoEnv.possibleActions)-1)

                        a1 = hfoEnv.possibleActions[action]

                        state2, reward, done, status, info = hfoEnv.step(a1)
                        episodeReward += reward
                        
                        #print(state2, reward, done, status, info)

                        y = computeTargets(reward, state2, gamma, done, target_value_network)

                        prediction = computePrediction(state1,action,value_network)

                        Y = torch.zeros(4)
                        Y[action] = y
                        Prediction = torch.zeros(4)
                        Prediction[action] = prediction

                        loss = loss_func(Y,Prediction)
                        loss.backward()

                        state1 = state2
                        t += 1
                        episodeSteps += 1
                        with lock:
                                counter.value = counter.value + 1

                        if done:

                                if status == GOAL:
                                        goal_buffer.append(1)
                                else:
                                        goal_buffer.append(0)

                                learning_str = "eval-" if args.eval else ""
                                logger.log_value('episode/' + learning_str + 'reward',
                                        episodeReward, episodeNumber)
                                logger.log_value('hyperparameters/epsilon', epsilon, episodeNumber)
                                logger.log_value('hyperparameters/lr', lr, episodeNumber)
                                for window in windows:
                                        logger.log_value(learning_str + "goals/%i" % window,
                                                        np.sum(goal_buffer[-window:]),
                                                        episodeNumber)
                                episodeNumber += 1
                                episodeReward = 0.0
                                episodeSteps  = 0
                                state1 = hfoEnv.reset()

                        if args.eval == False:
                                # TODO this will get some errors in the parallel implementation
                                # because once one worker updates it, the other wont, and 
                                # it can occur that a worker never gets to update it
                                if t % I_async == 0 or done or episodeNumber == episodes:
                                        # Async update of value_network using gradients

                                        with lock:
                                                # Add grads to value_network
                                                for param, shared_param in zip(
                                                            value_network.parameters(), 
                                                            target_value_network.parameters()):
                                                        shared_param._grad = param.grad
                                                        #value_network._grad = target_value_network.grad
                                                # Take a step
                                                optimizer.step(lr=lr)
                                                # Clean gradients
                                                optimizer.zero_grad()
                                        target_value_network.zero_grad()

                                if counter.value % I_tar == 0 or episodeNumber == episodes:
                                        # Update target network
                                        target_value_network.zero_grad()
                                        target_value_network.load_state_dict(value_network.state_dict())

                        hfoEnv.reset()

                if args.eval == False:
                        args.eval = True

        saveModelNetwork(value_network,"trained_models/params_last")
        # Finishing training and showing stats
        hfoEnv.quitGame()

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if done:
                y = torch.Tensor([reward]).to(device)
        else:
                qs = targetNetwork(nextObservation).data
                q_max = torch.max(qs)
                y = torch.Tensor([reward]).to(device) + discountFactor*q_max
        return y.detach()

def computePrediction(state, action, valueNetwork, possible_actions=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(state.shape) == 2 and state.shape[0] == 1:
            state = state[0]
        state = torch.Tensor(state)

        prediction = valueNetwork(state)#.to(device))
        prediction = prediction[action]

        return prediction
        
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

def loadModelNetwork(model, strDirectory):
        model.load_state_dict(torch.load(PATH))
