#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
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
		key = self._tuple(s) not in self._avg_pi.keys():
			self._avg_pi[key] = [1/len(self.possibleActions)]*len(self.possibleActions)
		return self._avg_pi[key]

	def pi(self, s):
		key = self._tuple(s) not in self._pi.keys():
			self._pi[key] = [1/len(self.possibleActions)]*len(self.possibleActions)
		return self._pi[key]

	def act(self):
		return self.possibleActions[np.argmax(self.pi(self._s1))]

	def calculateAveragePolicyUpdate(self):
		self._C[self._tuple(self._s1)] = self.C(self._s1) + 1
		C = self.C(self._s1)
		for i,a in enumerate(self.possibleActions):
			k = self._tuple(self._s1)
			self._avg_pi[k][i] = self.avg_pi(k)[i] + 1/C*(self.pi(k)[i] - self.avg_pi(k)[i])
		return self.avg_pi(k)# The avg policy in current state

	def calculatePolicyUpdate(self):

		return # The policy for the current state

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
		raise NotImplementedError

	def setLearningRate(self,lr):
		self._lr = lr
		
	def setWinDelta(self, winDelta):
		self._winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self._loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
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
