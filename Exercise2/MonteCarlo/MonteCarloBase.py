#!/usr/bin/env python3
# encoding utf-8

import numpy as np

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		
		self._gamma = discountFactor
		self._epsilon = epsilon
		self._ initVals = initVals

		self._Q  = {}
		self._pi = {} # a dict of dicts
		self._returns = {}
		self._episodes = []

	def learn(self):
		G = 0
		q_encountered = []
		for i,(s,a,r)in enumerate(self._episodes):
			if tuple([self._a,self.self._s1]) not in self._episodes[:i]:
				G = self._gamma*G + self._r
				self._add_G(s,a,G)
				q_sa = np.mean(self._returns[tuple([s,a])])
				self._Q[tuple([s,a])] = q_sa
				q_encountered.append(q_sa)

				opt_act = self._best_act(s)

				n = len(self.possibleActions)
				pi = []
				for act in self.possibleActions:
					if act == opt_act:
						pi.append(1-self._epsilon+self._epsilon/n)
					else:
						pi.append(self._epsilon/n)
				self._pi[tuple([s,a])] = pi

		return self._Q, q_encountered

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
		self._state_action_pairs.append(tuple([state,action,reward]))

	def setState(self, state):
		self._s1 = state

	def reset(self):
		self._episodes = []

	def act(self):
                '''
		Choice with parameters, similar to a multinomial dist
                '''
		pi_a = self.pi(self._s1)
		a = np.random.choice(self.possibleActions, size=1, p=pi_a.values())
                return a

	def pi(self, s):
		if tuple(s) not in self._pi.keys():
			n = len(self.possibleActions)
			self._pi[tuple(s)] = [self._epsilon/n]*n
		return self._pi[tuple(s)]

	def setEpsilon(self, epsilon):
		self._epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# TODO compute epsilon
		return tuple(self._epsilon)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
