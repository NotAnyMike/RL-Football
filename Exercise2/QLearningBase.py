#!/usr/bin/env python3
# encoding utf-8
import itertools
from time import time
import argparse

from pdb import set_trace
import numpy as np
from tensorboard_logger import configure, log_value

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
from hfo import GOAL

class QLearningAgent(Agent):
        def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
                super(QLearningAgent, self).__init__()

                self._LR = learningRate # Constant value
                self._lr = learningRate
                self._gamma = discountFactor
                self._epsilon  = epsilon
                self._EPSILON = epsilon
                self._Q = {}
                self._steps = 0
                self._episode = 0
                self._initVals = initVals

                self._min_epsilon = 0.01
                self._min_lr = 0.01

                self._s1 = None
                self._a  = None
                self._r  = None
                self._d  = None
                self._s2 = None

        def Q(self, state):
                if tuple(state) not in self._Q.keys():
                    self._Q[tuple(state)] = np.zeros(5)

                return self._Q[tuple(state)]
        
        def learn(self):
                Q = self.Q(self._s1)[self.possibleActions.index(self._a)]
                Q_max = self.Q(self._s2).max()
                TD_target = (self._r + self._gamma * Q_max)
                TD_delta = (TD_target - Q) 
                self._Q[tuple(self._s1)][self.possibleActions.index(self._a)] = Q + self._lr * TD_delta
                return TD_delta

        def act(self):
                '''
                Choose action based on e greedy policy
                '''
                if np.random.random() > self._epsilon: # Choose greedy action
                        a = self.possibleActions[self.Q(self._s1).argmax()]
                else: # Choose randomly
                        a = np.random.choice(self.possibleActions)
                return a

        def toStateRepresentation(self, state): 
                return tuple(state)

        def setState(self, state):
                self._s1 = state

        def setExperience(self, state, action, reward, status, nextState):
                self._s1 = state
                self._a  = action
                self._r  = reward
                self._d  = status # TODO We are not doing anything with this
                self._s2 = nextState

        def setLearningRate(self, learningRate):
                self._lr = learningRate
                
        def setEpsilon(self, epsilon):
                self._epsilon = epsilon

        def reset(self):
                self._epsilon = self._epsilon0
                #self._Q = {}
                #self._steps = 0
                #self._episode = 0

                self._s1 = None
                self._a  = None
                self._r  = None
                self._d  = None
                self._s2 = None
                
        def computeHyperparameters(self, numTakenActions, episodeNumber):
                self._episode = episodeNumber
                self._steps = numTakenActions

                # 1/x**2 decay
                #lr = self._LR / ((episodeNumber+1) ** (1/2) )
                #epsilon = (self._EPSILON - self._min_epsilon ) / ((episodeNumber+1) ** (1/2)) \
                #        + self._min_epsilon

                if episodeNumber > 100:
                    episodeNumber -= 100
                    epsilon = self._EPSILON - self._EPSILON / 300 * episodeNumber
                    lr = self._LR - self._LR / 300 * episodeNumber
                else: 
                    epsilon = self._EPSILON
                    lr = self._LR

                lr = lr if lr >= self._min_lr else self._min_lr
                epsilon = epsilon if epsilon >= self._min_epsilon else self._min_epsilon

                return lr, epsilon

if __name__ == '__main__':

        configure("tb/" + str(time()), flush_secs=5)

        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--numOpponents', type=int, default=0)
        parser.add_argument('--numTeammates', type=int, default=0)
        parser.add_argument('--numEpisodes', type=int, default=2000)

        args=parser.parse_args()

        # Initialize connection with the HFO server
        hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
        hfoEnv.connectToServer()

        # Initialize a Q-Learning Agent
        agent = QLearningAgent(learningRate = 0.95, discountFactor = 0.95, epsilon = 0.5)
        numEpisodes = args.numEpisodes

        rewards_buffer = []
        episode_length = []

        # Run training using Q-Learning
        numTakenActions = 0 
        history = [10,500]
        goals = [0]*max(history)
        for episode in range(numEpisodes):
                status = 0
                observation = hfoEnv.reset()
                cumulative_rewards = 0
                
                #while status==0:
                for t in itertools.count():
                        learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                        agent.setEpsilon(epsilon)
                        agent.setLearningRate(learningRate)
                        
                        obsCopy = observation.copy()
                        agent.setState(agent.toStateRepresentation(obsCopy))
                        action = agent.act()
                        numTakenActions += 1
                        
                        nextObservation, reward, done, status = hfoEnv.step(action)
                        agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
                        update = agent.learn()
                        
                        observation = nextObservation

                        cumulative_rewards += reward

                        if done:
                                if status == GOAL:
                                    goals.append(1)
                                else:
                                    goals.append(0)

                                rewards_buffer.append(cumulative_rewards)
                                episode_length.append(t)
                                log_value("episode/rewards", cumulative_rewards, episode)
                                log_value("episode/length", t, episode)
                                for h in history:
                                    log_value("training/goals-"+str(h), np.sum(goals[-h:]), episode)
                                log_value("training/lr", learningRate, episode)
                                log_value("training/epsilon", epsilon, episode)
                                break
        
        print(agent._Q.values())
