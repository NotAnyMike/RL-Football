import numpy as np
from copy import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self, discountRate):
		self.MDP = MDP()
		self._gamma = 0.99
		self.initVs()

	def initVs(self):
		self.v  = dict()
		self.pi = dict()
		for s in self.MDP.S:
			self.v[s] = 0
			self.pi[s] = copy(self.MDP.A)

	def BellmanUpdate(self):
		for s in self.MDP.S:
			max_val = 0
			opt_act = []
			for a in self.MDP.A:	
				probs = self.MDP.probNextStates(s,a)
				val = 0
				for a_prime,p in probs.items():
					r = self.MDP.getRewards(s,a,a_prime)
					val += p*(r+self._gamma*self.v[a_prime])
				if val > max_val:
					max_val = val
					opt_act = [a]
				elif val == max_val:
					opt_act.append(a)
			self.v[s]  = max_val
			self.pi[s] = opt_act
		return self.v, self.pi


def plot_value_and_policy(values, policy):
	
	data =np.zeros((5,5))

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.title('Value')
	for y in range(data.shape[0]):
		for x in range(data.shape[1]):
			data[y][x] = values[(x,y)]
			plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x], horizontalalignment='center',verticalalignment='center',)

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)

	plt.subplot(1, 2, 2)
	plt.title('Policy')
	for y in range(5):
		for x in range(5):
			for action in policy[(x,y)]:
				if action == 'DRIBBLE_UP':
					plt.annotate('',(x+0.5, y),(x+0.5,y+0.5),arrowprops={'width':0.1})
				if action == 'DRIBBLE_DOWN':
					plt.annotate('',(x+0.5, y+1),(x+0.5,y+0.5),arrowprops={'width':0.1})
				if action == 'DRIBBLE_RIGHT':
					plt.annotate('',(x+1, y+0.5),(x+0.5,y+0.5),arrowprops={'width':0.1})
				if action == 'DRIBBLE_LEFT':
					plt.annotate('',(x, y+0.5),(x+0.5,y+0.5),arrowprops={'width':0.1})
				if action == 'SHOOT':
					plt.text(x + 0.5, y + 0.5, action, horizontalalignment='center',verticalalignment='center',)

	heatmap = plt.pcolor(data)
	plt.gca().invert_yaxis()
	plt.colorbar(heatmap)
	plt.show()
		
if __name__ == '__main__':
	solution = BellmanDPSolver()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	plot_value_and_policy(values, policy)

	print("Values : ", values)
	print("Policy : ", policy)

