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

def train(idx, args, value_network, target_value_network, optimizer, lock, counter, 
                port, seed, I_tar, I_async, name=None):

        hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
        hfoEnv.connectToServer()

        if name == None:
                name = str(time())
        configure("tb/" + name, flush_secs=5)

        if args.eval:
                num_episodes = (args.eval_episodes)
        else:
                target_value_network.train()
                num_episodes = (args.episodes, args.eval_episodes)

        gamma = 0.99
        epsilon = 0.8
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
                        #train_epoch(epoch, args, model, device, train_loader, optimizer)
                        
                        if args.eval or np.random.random() <= epsilon:
                                qs = [computePrediction(state1,a,value_network) 
                                                for a in hfoEnv.possibleActions]
                                action = np.argmax(qs)
                                print("runing greedy") 
                        else:
                                action = random.randint(0,len(hfoEnv.possibleActions)-1)
                                print("running random") 

                        a1 = hfoEnv.possibleActions[action]

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

                        prediction = computePrediction(state1,a1,value_network)
                        loss = loss_func(prediction, y)
                        loss.backward()

                        state1 = state2
                        t += 1
                        episodeSteps += 1
                        with lock:
                                counter.value += counter.value + 1

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
                                                optimizer.step()
                                                # Clean gradients
                                                optimizer.zero_grad()
                                        target_value_network.zero_grad()

                                if counter.value % I_tar == 0 or episodeNumber == episodes:
                                        # Update target network
                                        target_value_network.zero_grad()
                                        target_value_network.load_state_dict(value_network.state_dict())

                        hfoEnv.reset()

                if args.eval == False:
                        args.eval == True

        # Finishing training and showing stats
        hfoEnv.quitGame()

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if done:
                y = torch.Tensor([reward]).to(device)
        else:
                qs = [computePrediction(nextObservation,a,targetNetwork) for a in range(4)]
                q_max = np.max(qs)
                y = torch.Tensor([reward]).to(device) + discountFactor*q_max
        return y.detach()

def computePrediction(state, action, valueNetwork, possible_actions=None):
        '''
        action should be a string
        '''
        if possible_actions == None:
                # to allow changing to different action space
                possible_actions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']
        if type(action) == str: 
                action = possible_actions.index(action)

        actions = [0]*len(possible_actions)
        actions[action] = 1

        inputs = np.concatenate([state,actions])
        inputs = torch.from_numpy(inputs)

        return valueNetwork(inputs.to('cuda'))
        
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

def loadModelNetwork(model, strDirectory):
        model.load_state_dict(torch.load(PATH))
