#!/usr/bin/env python3
# encoding utf-8
import itertools
import argparse
from time import time

import numpy as np
#from tensorboard_logger import configure, log_value

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
from hfo import GOAL

class MonteCarloAgent(Agent):
        def __init__(self, discountFactor, epsilon, initVals=0.0):
                super(MonteCarloAgent, self).__init__()
                
                self._gamma = discountFactor
                self._epsilon = epsilon
                self._initVals = initVals

                self._EPSILON = epsilon
                self._min_epsilon = 0.05

                self._Q  = {}
                self._pi = {} # a dict of dicts
                self._returns = {}

                self._S1 = []
                self._A  = []
                self._R  = []
                self._D  = []
                self._S2 = []

        def learn(self):
                G = 0
                self._q_encountered = []
                episode = list(zip(self._S1,self._A,self._R))
                pairs   = list(zip(self._S1,self._A))
                for i,(s,a,r) in reversed(list(enumerate(episode))):
                        G = self._gamma*G + r
                        if tuple([s,a]) not in pairs[:i]: # TODO be careful here
                                self._add_G(s,a,G) 
                                q_sa = np.mean(self._returns[tuple([s,a])])
                                self._Q[tuple([s,a])] = q_sa

                                opt_act = self._best_act(s)

                                n = len(self.possibleActions)
                                pi = []
                                for act in self.possibleActions:
                                        if act == opt_act:
                                                pi.append(1.0-self._epsilon+self._epsilon/n)
                                        else:
                                                pi.append(self._epsilon/n)
                                self._pi[tuple([s,a])] = tuple(pi)
                                
                                #self._q_encountered.append(q_sa)
                                self._q_encountered.insert(0,q_sa)

                return self._Q, self._q_encountered

        def _best_act(self, state):
                max_val = None # To allow negative values
                opt_act = []
                for a in self.possibleActions:
                    q = self.Q(state, a)
                    if max_val == None or q > max_val:
                        opt_act = [a]
                        max_val = q
                    elif q == max_val:
                        opt_act.append(a) 
                a = np.random.choice(opt_act)
                return a

        def Q(self, state, action):
            if tuple([state,action]) not in self._Q.keys():
                self._Q[tuple([state,action])] = 0.0

            return self._Q[tuple([state,action])]
                                
        def _add_G(self,s,a,G):
                if tuple([s,a]) not in self._returns.keys():
                        self._returns[tuple([s,a])] = []
                self._returns[tuple([s,a])].append(G)
        
        def toStateRepresentation(self, state):
                return tuple(state)

        def setExperience(self, state, action, reward, status, nextState):
                self._S1.append(tuple(state))
                self._A.append(action)
                self._R.append(reward)
                self._D.append(status)
                self._S2.append(tuple(nextState))

        def setState(self, state):
                self._s1 = state

        def reset(self):
                self._q_encountered = []
                self._S1 = []
                self._A  = []
                self._R  = []
                self._D  = []
                self._S2 = []

        def act(self):
                '''
                Choice with parameters, similar to a multinomial dist
                '''
                pi_a = self.pi(self._s1)
                print(pi_a.values())
                a = np.random.choice(self.possibleActions, p=list(pi_a.values()))
                return a

        def pi(self, s):
                if tuple(s) not in self._pi.keys():
                        n = len(self.possibleActions)
                        self._pi[tuple(s)] = dict([[a,1.0/n] for a in self.possibleActions])
                return self._pi[tuple(s)]

        def setEpsilon(self, epsilon):
                self._epsilon = epsilon

        def computeHyperparameters(self, numTakenActions, episodeNumber):
                # TODO compute epsilon
                self._episode = episodeNumber
                self._steps = numTakenActions

                if episodeNumber > 100:
                    episodeNumber -= 100
                    epsilon = self._EPSILON - self._EPSILON / 200 * episodeNumber
                else: 
                    epsilon = self._EPSILON

                epsilon = epsilon if epsilon >= self._min_epsilon else self._min_epsilon

                return epsilon


if __name__ == '__main__':

        #configure("tb/MC" + str(time()))

        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--numOpponents', type=int, default=0)
        parser.add_argument('--numTeammates', type=int, default=0)
        parser.add_argument('--numEpisodes', type=int, default=1000)

        args=parser.parse_args()

        #Init Connections to HFO Server
        hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
        hfoEnv.connectToServer()

        rewards_buffer = []
        episode_length = []
        history = [10,500]
        goals = [0]*max(history)

        # Initialize a Monte-Carlo Agent
        agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
        numEpisodes = args.numEpisodes
        numTakenActions = 0
        # Run training Monte Carlo Method
        for episode in range(numEpisodes):      
                agent.reset()
                observation = hfoEnv.reset()
                status = 0
                cumulative_rewards = 0

                for t in itertools.count():
                        epsilon = agent.computeHyperparameters(numTakenActions, episode)
                        agent.setEpsilon(epsilon)
                        obsCopy = observation.copy()
                        agent.setState(agent.toStateRepresentation(obsCopy))
                        action = agent.act()
                        numTakenActions += 1
                        nextObservation, reward, done, status = hfoEnv.step(action)
                        agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
                        observation = nextObservation
                        cumulative_rewards += reward

                        if done:
                                if status == GOAL:
                                    goals.append(1)
                                else:
                                    goals.append(0)

                                rewards_buffer.append(cumulative_rewards)
                                episode_length.append(t)
                                #log_value("episode/rewards", cumulative_rewards, episode)
                                #log_value("episode/length", t, episode)
                                #for h in history:
                                    #log_value("training/goals-"+str(h), np.sum(goals[-h:]), episode)
                                #log_value("training/epsilon", epsilon, episode)
                                break

                agent.learn()
