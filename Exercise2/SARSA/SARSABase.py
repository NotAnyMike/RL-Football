#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class SARSAAgent(Agent): 
        def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
                super(SARSAAgent, self).__init__()

                self._lr = learningRate
                self._gamma = discountFactor
                self._epsilon = epsilon
                self._initVals = initVals

                self._Q = {}

        def Q(state, action):
            if tuple([state,action]) not in self._Q.keys():
                self._q[tuple([state,action])] = 0.0
            return self._q[tuple([state,action])]

        def learn(self):
                q  = self.Q(self._s1, self._a)
                a2 = self._best_act(self._s2)
                q2 = self.Q(self._s2, a2)
                delta = self._lr (self._r + self._gamma * q2 - q)
                self._Q[tuple([self._s1, self._a])] = q + delta
                return delta

        def best_act(state):
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

        def act(self):
                '''
                Choose the best action and return it
                '''
                p = self._epsilon #+ (self._epsilon/len(self.possibleActions))
                case = np.random.binomial(1, p, 1)
                if case: # Choose greedy action
                        a = self.best_act(self._s1)
                else: # Choose randomly
                        a = np.random.choice(self.possibleActions)
                return a

        def setState(self, state):
                self._s1 = state

        def setExperience(self, state, action, reward, status, nextState):
                self._s1 = state
                self._a  = action
                self._r  = reward
                self._d  = status
                print("status: ", status)
                self._s2 = nextState

        def computeHyperparameters(self, numTakenActions, episodeNumber):
                return self._lr, self._epsilon # TODO

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

        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--numOpponents', type=int, default=0)
        parser.add_argument('--numTeammates', type=int, default=0)
        parser.add_argument('--numEpisodes', type=int, default=500)

        args=parser.parse_args()
        
        numEpisodes = args.numEpisodes
        # Initialize connection to the HFO environment using HFOAttackingPlayer
        hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
        hfoEnv.connectToServer()

        # Initialize a SARSA Agent
        agent = SARSAAgent(0.1, 0.99)

        # Run training using SARSA
        numTakenActions = 0 
        for episode in range(numEpisodes):      
                agent.reset()
                status = 0

                observation = hfoEnv.reset()
                nextObservation = None
                epsStart = True

                while status==0:
                        learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                        agent.setEpsilon(epsilon)
                        agent.setLearningRate(learningRate)
                        
                        obsCopy = observation.copy()
                        agent.setState(agent.toStateRepresentation(obsCopy))
                        action = agent.act()
                        numTakenActions += 1

                        nextObservation, reward, done, status = hfoEnv.step(action)
                        print(obsCopy, action, reward, nextObservation)
                        agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
                        
                        if not epsStart :
                                agent.learn()
                        else:
                                epsStart = False
                        
                        observation = nextObservation

                agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
                agent.learn()

        print(agent._Q.values())

        
