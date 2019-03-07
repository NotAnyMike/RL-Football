#!/usr/bin/env python3
# encoding utf-8
import itertools
import argparse
from time import time

import numpy as np
from tensorboard_logger import configure, log_value

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent 
from hfo import GOAL 

class SARSAAgent(Agent): 
        def __init__(self, learningRate, discountFactor, epsilon=1, initVals=0.0):
                super(SARSAAgent, self).__init__()

                self._lr = learningRate
                self._gamma = discountFactor
                self._epsilon = epsilon
                self._initVals = initVals

                self._LR = learningRate # Constant value
                self._EPSILON = epsilon
                self._min_epsilon = 0.01
                self._min_lr = 0.01

                self._Q = {}

        def Q(self, state):
            if tuple(state) not in self._Q.keys():
                self._Q[tuple(state)] = [0.0]*len(self.possibleActions)

            return self._Q[tuple(state)]

        def learn(self):

                q1_vals = self.Q(self._s1)
                q1 = q1_vals[self.possibleActions.index(self._a)]
                
                q2_vals = self.Q(self._s2)
                a2 = self.policy(q2_vals)
                TD_target = self._r + self._gamma * q2_vals[self.possibleActions.index(a2)]

                TD_delta  = TD_target - q1
                
                self._Q[tuple(self._s1)][self.possibleActions.index(self._a)] = q1 + self._lr * TD_delta
                return TD_delta

        def act(self):
                '''
                Choose action based on policy
                '''
                q = self.Q(self._s1)
                return self.policy(q)

        def policy(self, q_values):
                actions_prob = np.zeros(len(self.possibleActions)) + \
                        self._epsilon / len(self.possibleActions)
                actions_prob[np.argmax(q_values)] += 1. - self._epsilon
                a = np.random.choice(self.possibleActions, p=actions_prob)
                return a


        def setState(self, state):
                self._s1 = state

        def setExperience(self, state, action, reward, status, nextState):
                self._s1 = state
                self._a  = action
                self._r  = reward
                self._d  = status
                self._s2 = nextState

        def computeHyperparameters(self, numTakenActions, episodeNumber):
                #return self._lr, self._epsilon # TODO
                self._episode = episodeNumber
                self._steps = numTakenActions

                # 1/x**2 decay
                #lr = self._LR / ((episodeNumber+1) ** (1/2) )
                #epsilon = (self._EPSILON - self._min_epsilon ) / ((episodeNumber+1) ** (1/2)) \
                #        + self._min_epsilon

                if episodeNumber > 100:
                    episodeNumber -= 100
                    epsilon = self._EPSILON - self._EPSILON / 200 * episodeNumber
                    lr = self._LR - self._LR / 400 * episodeNumber
                else: 
                    epsilon = self._EPSILON
                    lr = self._LR

                lr = lr if lr >= self._min_lr else self._min_lr
                epsilon = epsilon if epsilon >= self._min_epsilon else self._min_epsilon

                return lr, epsilon

        def toStateRepresentation(self, state):
                return tuple(state)

        def reset(self):
                self._s1 = None
                self._a  = None
                self._r  = None
                self._d  = None
                self._s2 = None

        def setLearningRate(self, learningRate):
                self._lr = learningRate

        def setEpsilon(self, epsilon):
                self._epsilon = epsilon

if __name__ == '__main__':

        configure("tb/sarsa" + str(time()))

        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--numOpponents', type=int, default=0)
        parser.add_argument('--numTeammates', type=int, default=0)
        parser.add_argument('--numEpisodes', type=int, default=1000)

        args=parser.parse_args()
        
        numEpisodes = args.numEpisodes
        # Initialize connection to the HFO environment using HFOAttackingPlayer
        hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
        hfoEnv.connectToServer()

        # To keep track
        rewards_buffer = []
        episode_length = []
        history = [10,500]
        goals = [0]*max(history)

        # Initialize a SARSA Agent
        agent = SARSAAgent(learningRate=0.95, discountFactor=0.95, epsilon=1)

        # Run training using SARSA
        numTakenActions = 0 
        for episode in range(numEpisodes):      
                agent.reset()
                status = 0

                observation = hfoEnv.reset()
                nextObservation = None
                cumulative_rewards = 0

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

                        cumulative_rewards += reward
                        observation = nextObservation

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
                        

                #agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
                #agent.learn()

        print(agent._Q.values())

        
