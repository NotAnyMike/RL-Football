#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
from tensorboard_logger import configure, log_value
import numpy as np
from pdb import set_trace
from datetime import datetime
import itertools
import argparse
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()
		self._lr = learningRate
		self._gamma = discountFactor
		self._epsilon  = epsilon
		self._Q = {}
		self._C = {}
		self._N = {}
		self._steps = 0
		self._episode = 0
		self._initVals = initVals

		self._LR = learningRate # Constant value
		self._EPSILON = epsilon
		self._min_epsilon = 0.1
		self._min_lr = 0.01

		self._s1 = None
		self._a1 = None
		self._a2 = None
		self._r  = None
		self._d  = None
		self._s2 = None

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self._s1 = state
		self._a1 = action
		self._a2 = oppoActions
		self._r  = reward
		self._d  = status # TODO We are not doing anything with this
		self._s2 = nextState
		
	def Q(self, state, a1, a2):
		key = self._tuple([state, a1,a2])
		if key not in self._Q.keys():
			#self._Q[tuple(state)] = np.zeros(len(self.possibleActions))
			self._Q[key] = 0.0
			return 0.0
		return self._Q[key]

	def C(self, state, a2):
		key = self._tuple([state,a2])
		if key not in self._C.keys():
			self._C[key] = 0.0
			return 0.0
		return self._C[key]

	def N(self, state):
		key = self._tuple(state)
		if key not in self._N.keys():
			self._N[key] = 0.0
			return 0.0
		return self._N[key]

	def _V(self, s, a):
		n = self.N(s)
		if n == 0.0:
			vals = [self.Q(s,a,a2)/len(self.possibleActions) \
					for a2 in self.possibleActions]
		else:
			vals = [self.C(s,a2)/n*self.Q(s,a,a2) for a2 in self.possibleActions]
		return sum(vals)
	
	def learn(self):
		# Updating Q
		V_max = max([self._V(self._s2,a) for a in self.possibleActions])
		Q = self.Q(self._s1, self._a1, self._a2)
		TD_target = (self._r + self._gamma * V_max)
		TD_delta = (TD_target - Q) 
		self._Q[self._tuple([self._s1,self._a1,self._a2])] = Q + self._lr * TD_delta

		# Updating C
		self._C[self._tuple([self._s1,self._a2])] = self.C(self._s1,self._a2)+1
		#print("this value should be increasing",self._C[self._tuple([self._s1,self._a2])])

		# Updating N
		self._N[self._tuple(self._s1)] = self.N(self._s1)+1
		#rint("This value should also increase", self._N[self._tuple(self._s1)])

		return TD_delta*self._lr

	def act(self):
		'''
		Choose action based on e greedy policy
		Given that we are using a q table is allright to not use onehot
		'''
		if np.random.random() > self._epsilon: # Choose greedy action
			vec = [self._V(self._s1, a1) for a1 in self.possibleActions]
			a = self.possibleActions[np.argmax(vec)]
		else: # Choose randomly
			a = np.random.choice(self.possibleActions)
		return a

	def setEpsilon(self, epsilon) :
		self._epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self._lr = learningRate

	def setState(self, state):
		self._s1 = state

	def _tuple(self, elems):
		if type(elems) == list or type(elems) == tuple:
			l = ()
			for x in elems:
				l = l+self._tuple(x)
			return l
		else:
			return tuple([elems])
				

	def toStateRepresentation(self, rawState):
		return self._tuple(rawState)
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self._episode = episodeNumber
		self._steps = numTakenActions

		delay = 0
		if episodeNumber > delay:
			epsilon = self._EPSILON - self._EPSILON / 15000 * (episodeNumber-delay)
		else: 
			epsilon = self._EPSILON

		delay = 0
		if episodeNumber > delay:
			lr = self._LR - self._LR / 15000 * (episodeNumber-delay)
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
		configure("tb/JAL" + str(datetime.now()))
	#####################################################

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, 
			numAgents = args.numAgents, visualize=args.visualize)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.9, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

			if args.visualize: MARLEnv.visualizeState(reward)

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
