#!/usr/bin/env python3
# encoding utf-8
import itertools

from pdb import set_trace
import numpy as np
from tensorboard_logger import configure, log_value

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class QLearningAgent(Agent):
        def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
                super(QLearningAgent, self).__init__()

                self._lr = learningRate
                self._gamma = discountFactor
                self._epsilon  = epsilon
                self._epsilon0 = epsilon
                self._Q = {}
                self._steps = 0
                self._episode = 0
                self._initVals = initVals

                self._s1 = None
                self._a  = None
                self._r  = None
                self._d  = None
                self._s2 = None

        def Q(self, state, action):
                if tuple([state,action]) not in self._Q.keys():
                    self._Q[tuple([state,action])] = 0.0

                return self._Q[tuple([state,action])]
        
        def learn(self):
                Q = self.Q(self._s1, self._a)
                all_q = [self.Q(self._s2, pos_a) for pos_a in self.possibleActions]     
                Q_max = max(all_q)
                delta = self._lr * (self._r + self._gamma * Q_max - Q)
                #self.Q(self._s1, self._a) = self.Q(self._s1,self._a) + delta
                self._Q[tuple([self._s1, self._a])] = Q + delta
                return delta

        def act(self):
                '''
                Choose the best action and return it
                '''
                if np.random.random() <= self._epsilon: # Choose greedy action
                        max_val = None # To allow negative values
                        opt_act = []
                        for a in self.possibleActions:
                            q = self.Q(self._s1, a)
                            if max_val == None or q > max_val:
                                opt_act = [a]
                                max_val = q
                            elif q == max_val:
                                opt_act.append(a) 
                        a = np.random.choice(opt_act)
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
                # TODO implement a schedule for epsilon and lr (more important)
                self._episode = episodeNumber
                self._steps = numTakenActions
                return self._lr, self._epsilon

if __name__ == '__main__':

        configure("tb", flush_secs=5)

        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--numOpponents', type=int, default=0)
        parser.add_argument('--numTeammates', type=int, default=0)
        parser.add_argument('--numEpisodes', type=int, default=500)

        args=parser.parse_args()

        # Initialize connection with the HFO server
        hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
        hfoEnv.connectToServer()

        # Initialize a Q-Learning Agent
        agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
        numEpisodes = args.numEpisodes

        rewards_buffer = []
        episode_length = []

        # Run training using Q-Learning
        numTakenActions = 0 
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
                                rewards_buffer.append(cumulative_rewards)
                                episode_length.append(t)
                                log_value("episode_rewards", cumulative_rewards, episode)
                                log_value("episode_length", t, episode)
                                break
        
        print(agent._Q.values())
