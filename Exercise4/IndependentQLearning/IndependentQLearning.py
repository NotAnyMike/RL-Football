#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from tensorboard_logger import configure, log_value
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
from time import time
import numpy as np
from pdb import set_trace
from hfo import GOAL
import argparse
from datetime import datetime
                
class IndependentQLearningAgent(Agent):
        def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
                super(IndependentQLearningAgent, self).__init__()

                self._lr = learningRate
                self._gamma = discountFactor
                self._epsilon  = epsilon
                self._Q = {}
                self._steps = 0
                self._episode = 0
                self._initVals = initVals

                self._LR = learningRate # Constant value
                self._EPSILON = epsilon
                self._min_epsilon = 0.1
                self._min_lr = 0.01

                self._s1 = None
                self._a  = None
                self._r  = None
                self._d  = None
                self._s2 = None

        def setExperience(self, state, action, reward, status, nextState):
                self._s1 = state
                self._a  = action
                self._r  = reward
                self._d  = status # TODO We are not doing anything with this
                self._s2 = nextState

        def Q(self, state):
                if tuple(state) not in self._Q.keys():
                    self._Q[tuple(state)] = np.zeros(len(self.possibleActions))

                return self._Q[tuple(state)]
        
        def learn(self):
                Q = self.Q(self._s1)[self.possibleActions.index(self._a)]
                Q_max = self.Q(self._s2).max()
                TD_target = (self._r + self._gamma * Q_max)
                TD_delta = (TD_target - Q) 
                self._Q[tuple(self._s1)][self.possibleActions.index(self._a)] = Q + self._lr * TD_delta
                return TD_delta*self._lr

        def act(self):
                '''
                Choose action based on e greedy policy
                Given that we are using a q table is allright to not use onehot
                '''
                if np.random.random() > self._epsilon: # Choose greedy action
                        a = self.possibleActions[self.Q(self._s1).argmax()]
                else: # Choose randomly
                        a = np.random.choice(self.possibleActions)
                return a

        def toStateRepresentation(self, state):
                if type(state[0]) == list:
                    state = [self.toStateRepresentation(x) for x in state]
                return tuple(state)

        def setState(self, state):
                self._s1 = state

        def setEpsilon(self, epsilon):
                self._epsilon = epsilon
                
        def setLearningRate(self, learningRate):
                self._lr = learningRate
                
        def computeHyperparameters(self, numTakenActions, episodeNumber):
                self._episode = episodeNumber
                self._steps = numTakenActions

                delay = 0
                if episodeNumber > delay:
                    epsilon = self._EPSILON - self._EPSILON / 10000 * (episodeNumber-delay)
                else: 
                    epsilon = self._EPSILON

                delay = 100
                if episodeNumber > delay:
                    lr = self._LR - self._LR / 10000 * (episodeNumber-delay)
                else: 
                    lr = self._LR

                lr = lr if lr >= self._min_lr else self._min_lr
                epsilon = epsilon if epsilon >= self._min_epsilon else self._min_epsilon

                return lr, epsilon

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--numOpponents', type=int, default=1)
        parser.add_argument('--numAgents', type=int, default=2)
        parser.add_argument('--numEpisodes', type=int, default=50000)
        parser.add_argument('--visualize', type=bool, default=False)

        args=parser.parse_args()

        ########### with debugging purposes only ############
        debug = True
        if debug:
                rewards_buffer = []
                history = [10,500]
                goals = [0]*max(history)
                configure("tb/IQL" + str(datetime.now()))
        #####################################################

        MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents,
                visualize=args.visualize)
        agents = []
        for i in range(args.numAgents):
                agent = IndependentQLearningAgent(learningRate = 0.99, discountFactor = 0.9, epsilon = 1.0)
                agents.append(agent)

        numEpisodes = args.numEpisodes
        numTakenActions = 0
        for episode in range(numEpisodes):      
                status = ["IN_GAME","IN_GAME","IN_GAME"]
                observation = MARLEnv.reset()
                totalReward = 0.0
                timeSteps = 0
                        
                while status[0]=="IN_GAME":
                        for agent in agents:
                                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                                agent.setEpsilon(epsilon)
                                agent.setLearningRate(learningRate)
                        actions = []
                        #stateCopies = []
                        stateCopies, nextStateCopies = [], []
                        for agentIdx in range(args.numAgents):
                                obsCopy = deepcopy(observation[agentIdx])
                                stateCopies.append(obsCopy)
                                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                                actions.append(agents[agentIdx].act())
                        numTakenActions += 1
                        nextObservation, reward, done, status = MARLEnv.step(actions)

                        if args.visualize: MARLEnv.visualizeState(reward)

                        for agentIdx in range(args.numAgents):
                                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
                                        status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                                agents[agentIdx].learn()
                                
                        observation = nextObservation
                        
                        ############# with debugging purposes ############
                        if done[0] and debug:
                                if status[0] == "GOAL":
                                    goals.append(1)
                                else:
                                    goals.append(0)

                                #rewards_buffer.append(cumulative_rewards)
                                #log_value("episode/rewards", cumulative_rewards, episode)
                                for h in history:
                                    log_value("training/goals-"+str(h), np.sum(goals[-h:]), episode)
                                log_value("training/lr", learningRate, episode)
                                log_value("training/epsilon", epsilon, episode)
                        ##################################################
