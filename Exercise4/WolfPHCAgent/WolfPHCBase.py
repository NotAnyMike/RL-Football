#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
from datetime import datetime
from tensorboard_logger import configure, log_value
import numpy as np
                
class WolfPHCAgent(Agent):
        def __init__(self, learningRate, discountFactor, winDelta=0.01, 
                    loseDelta=0.1, initVals=0.0, epsilon=0.0):
                super(WolfPHCAgent, self).__init__()

                self._lr = learningRate
                self._gamma = discountFactor
                self._winDelta = winDelta
                self._loseDelta = loseDelta

                self._avg_pi = {}
                self._pi = {}
                self._Q  = {}
                self._n  = {}
                self._C  = {}

                # Setting max and mins
                self._max_winDelta = 0.9
                self._min_winDelta = winDelta

                self._max_loseDelta = 0.9
                self._min_loseDelta = loseDelta

                self._max_epsilon = 0.9
                self._min_epsilon = epsilon

                self._max_lr = 0.9
                self._min_lr = learningRate
                
        def setExperience(self, state, action, reward, status, nextState):
                self._s1 = state
                self._a  = action
                self._r  = reward
                self._d  = status
                self._s2 = nextState

        def learn(self):
                Q = self.Q(self._s1, self._a)
                Q_max = max([self.Q(self._s2, a2) for a2 in self.possibleActions])

                TD_target = self._r + self._gamma * Q_max
                TD_delta = TD_target - Q

                self._Q[self._tuple([self._s1, self._a])] = Q + self._lr * TD_delta
                return TD_delta*self._lr # Return the delta in Q

        def avg_pi(self, s):
                key = self._tuple(s) 
                if key not in self._avg_pi.keys():
                        self._avg_pi[key] = [1/len(self.possibleActions)]*len(self.possibleActions)
                return self._avg_pi[key]

        def pi(self, s):
                key = self._tuple(s) 
                if key not in self._pi.keys():
                        self._pi[key] = [1/len(self.possibleActions)]*len(self.possibleActions)
                return self._pi[key]

        def Q(self,s,a):
                key = self._tuple([s,a])
                if key not in self._Q.keys():
                        self._Q[key] = 0.0
                        return 0.0
                return self._Q[key]

        def C(self,s):
                key = self._tuple(s)
                if key not in self._C.keys():
                        self._C[key] = 0.0
                        return 0.0
                return self._C[key]

        def act(self):
                '''
                Choose action based on e greedy policy
                '''
                if np.random.random() > self._epsilon: # Choose greedy action
                        #a = self.possibleActions[self.Q(self._s1).argmax()]
                        #a = self.possibleActions[np.argmax(self.pi(self._s1))]
                        a = np.random.choice(self.possibleActions,p=self.pi(self._s1))
                else: # Choose randomly
                        a = np.random.choice(self.possibleActions)
                return a

        def calculateAveragePolicyUpdate(self):
                self._C[self._tuple(self._s1)] = self.C(self._s1) + 1
                C = self.C(self._s1)
                for i,a in enumerate(self.possibleActions):
                        k = self._tuple(self._s1)
                        self._avg_pi[k][i] = self.avg_pi(k)[i] + 1/C*(self.pi(k)[i] - self.avg_pi(k)[i])
                return self.avg_pi(k)# The avg policy in current state

        def calculatePolicyUpdate(self):
                # Find the suboptimal actions
                Q_max = max([self.Q(self._s1,a) for a in self.possibleActions])
                actions_sub = [a for a in self.possibleActions if self.Q(self._s1,a) < Q_max] # TODO check this maybe it has an error
                assert len(actions_sub) != len(self.possibleActions)

                # Decide which lr to use
                qs = [self.Q(self._s1,a) for a in self.possibleActions]
                sum_avg  = np.dot(self.avg_pi(self._s1),qs) 
                sum_norm = np.dot(self.pi(self._s1),qs)
                delta = self._winDelta if sum_norm >= sum_avg else self._loseDelta

                # Update probability of suboptimal actions
                p_moved = 0.0
                for i,a in enumerate(self.possibleActions):
                        pi = self.pi(self._s1)
                        if a in actions_sub:
                                p_moved = p_moved + min([delta/len(actions_sub), pi[i]])
                                self._pi[self._tuple(self._s1)][i] = pi[i] - min([delta/len(actions_sub), pi[i]])

                        # Update prob of optimal actions
                for i,a in enumerate(self.possibleActions):
                        pi = self.pi(self._s1)
                        if a not in actions_sub:
                                self._pi[self._tuple(self._s1)][i] = pi[i] + p_moved/\
                                                (len(self.possibleActions) - len(actions_sub))

                return self.pi(self._s1) # The policy for the current state

        def _tuple(self, args):
                if type(args) == list or type(args) == tuple:
                        t = ()
                        for x in args:
                                t = t + self._tuple(x)
                        return t
                else:
                        return tuple([args])
        
        def toStateRepresentation(self, state):
                return self._tuple(state)

        def setState(self, state):
                self._s1 = state

        def setLearningRate(self,lr):
                self._lr = lr
                
        def setWinDelta(self, winDelta):
                self._winDelta = winDelta
                
        def setLoseDelta(self, loseDelta):
                self._loseDelta = loseDelta
        
        def computeHyperparameters(self, numTakenActions, episodeNumber):
                # TODO something to do here
                self._episode = episodeNumber
                self._steps = numTakenActions

                pices = 20000
                delay = 0
                if episodeNumber > delay:
                    epsilon = self._max_epsilon - self._max_epsilon / pices * (episodeNumber-delay)
                else: 
                    epsilon = self._max_epsilon

                delay = 0
                #pices = 5000
                if episodeNumber > delay:
                    lr = self._max_lr - self._max_lr / pices * (episodeNumber-delay)
                else: 
                    lr = self._max_lr

                delay = 0
                if episodeNumber > delay:
                    winDelta = self._max_winDelta - self._max_winDelta / pices * (episodeNumber-delay)
                else: 
                    winDelta = self._max_winDelta

                delay = 0
                if episodeNumber > delay:
                    loseDelta = self._max_loseDelta - self._max_loseDelta / pices * (episodeNumber-delay)
                else: 
                    loseDelta = self._max_loseDelta

                loseDelta = loseDelta if loseDelta >= self._min_loseDelta else self._min_loseDelta
                winDelta = winDelta if winDelta >= self._min_winDelta else self._min_winDelta
                lr = lr if lr >= self._min_lr else self._min_lr
                epsilon = epsilon if epsilon >= self._min_epsilon else self._min_epsilon

                self._epsilon = epsilon

                return loseDelta, winDelta, lr

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--numOpponents', type=int, default=1)
        parser.add_argument('--numAgents', type=int, default=2)
        parser.add_argument('--numEpisodes', type=int, default=100000)
        parser.add_argument('--visualize', type=bool, default=False)

        args=parser.parse_args()

        ########### with debugging purposes only ############
        debug = True
        if debug:
                rewards_buffer = []
                history = [10,500]
                goals = [0]*max(history)
                configure("tb/WOLF" + str(datetime.now()))
        #####################################################

        numOpponents = args.numOpponents
        numAgents = args.numAgents
        MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

        agents = []
        for i in range(args.numAgents):
                agent = WolfPHCAgent(learningRate = 0.01, discountFactor = 0.99, 
                        winDelta=0.01, loseDelta=0.1)
                agents.append(agent)

        numEpisodes = args.numEpisodes
        numTakenActions = 0
        for episode in range(numEpisodes):      
                status = ["IN_GAME","IN_GAME","IN_GAME"]
                observation = MARLEnv.reset()
                
                while status[0]=="IN_GAME":
                        for agent in agents:
                                loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
                                agent.setLoseDelta(loseDelta)
                                agent.setWinDelta(winDelta)
                                agent.setLearningRate(learningRate)
                        actions = []
                        perAgentObs = []
                        agentIdx = 0
                        for agent in agents:
                                obsCopy = deepcopy(observation[agentIdx])
                                perAgentObs.append(obsCopy)
                                agent.setState(agent.toStateRepresentation(obsCopy))
                                actions.append(agent.act())
                                agentIdx += 1
                        nextObservation, reward, done, status = MARLEnv.step(actions)
                        numTakenActions += 1

                        agentIdx = 0
                        for agent in agents:
                                agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
                                        status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                                agent.learn()
                                agent.calculateAveragePolicyUpdate()
                                agent.calculatePolicyUpdate()
                                agentIdx += 1
                        
                        observation = nextObservation

                        ############# with debugging purposes ############
                        if args.visualize: MARLEnv.visualizeState(reward)

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
                                log_value("training/winDelta", winDelta, episode)
                                log_value("training/loseDelta", loseDelta, episode)
                                log_value("training/epsilon", agents[0]._epsilon, episode)
                        ##################################################
