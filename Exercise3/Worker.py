from pdb import set_trace
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tensorboard_logger import configure, log_value

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train(idx, args, value_network, target_value_network, optimizer, lock, counter, port, seed, I_tar, I_async):

        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
        hfoEnv.connectToServer()

        print(hfoEnv.hfo.getStateSize())

        episodeNumber = 0
        episodeReward = 0
        episodeSteps  = 0
        gamma = 0.99
        epsilon = 0.8
        t = 0
        loss_func = nn.MSELoss()
        state1 = hfoEnv.reset()
        configure("tb/" + str(time()), flush_secs=5)
        #for epoch in range(1, args.epochs + 1):
        while episodeNumber < args.episodes:
                #train_epoch(epoch, args, model, device, train_loader, optimizer)
                
                if args.eval or np.random.random() < epsilon:
                        qs = [computePrediction(state1,a,value_network) for a in range(4)]
                        action = np.argmax(qs)
                        print("runing greedy") 
                else:
                        action = random.randint(0,3)
                        print("runing random") 

                a1 = hfoEnv.possibleActions[action]
                a1_num = hfoEnv.possibleActions.index(a1)

                state2, reward, done, status, info = hfoEnv.step(a1)
                episodeReward += reward
                
                #print(state2, reward, done, status, info)

                if done:
                        learning_str = "eval" if args.eval else "learning"
                        log_value('episode reward %s (idx %i)' % (learning_str,idx), episodeReward, episodeNumber)
                        episodeNumber += 1
                        episodeReward = 0.0
                        episodeSteps  = 0
                        hfoEnv.reset()

                y = computeTargets(reward, state2, gamma, done, target_value_network)

                prediction = computePrediction(state1,a1_num, value_network)
                loss = loss_func(prediction, y)
                loss.backward()

                state1 = state2
                t += 1
                episodeSteps += 1
                with lock:
                        counter.value += counter.value + 1

                if args.eval == False:
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

        # Finishing training and showing stats
        hfoEnv.quitGame()

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

