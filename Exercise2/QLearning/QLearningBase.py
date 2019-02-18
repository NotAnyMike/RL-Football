#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()

		self._lr = learningRate
		self._discountFactor = DiscountFactor
		self._epsilon  = epsilon
		self._epsilon0 = epsilon
		self._Q = {}
		self._steps = 0
		self._episode = 0

		self._s1 = None
		self._a  = None
		self._r  = None
		self._d  = None
		self._s2 = None

	def learn(self):
		Q = self._Q[([self._s, self._a])]
		all_q = [self._Q[([self._s, pos_a])] for pos_a in self.possibleActions]	
		Q_max = max(all_q)
		delta = self._lr * (self._r + self._gamma * Q_max - Q)
		self._Q[([self._s, self._a])] +=  delta
		return delta

	def act(self):
		'''
		Choose an action and return it
		'''
		p = self._epsilon #+ (self._epsilon/len(self.possibleActions))
		case = np.random.binomial(1, p, 1)
		if case: # Choose greedy action
			a = max(self._Q, key=self._Q.get)
		else: # Choose randomly
			a = np.random.choice(self.possibleActions)
		return a

	def toStateRepresentation(self, state):
		return state

	def setState(self, state):
		if not state in list(zip(*self.Q.keys())[0]):
			for a in self.possibleActions:
				self._Q[(state,a)] = 0
		self._s1 = state

	def setExperience(self, state, action, reward, status, nextState):
		self._s1 = state
		self._a  = action
		self._r  = reward
		self._d  = status
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
		# TODO
		self._episode = episodeNumber
		self._steps = numTakenActions
		return self._lr, self._epsilon

if __name__ == '__main__':

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

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
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
	
