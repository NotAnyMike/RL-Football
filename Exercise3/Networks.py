import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self):
		super(ValueNetwork, self).__init__()
                hidden_sizes = [50, 50]
                input_dim = 50
                self.layers = []

                for size in range(hidden_sizes):
                        self.layers.add(nn.Linear(input_dim, size))
                        input_dim = size
                
	def forward(self, inputs) :
                for layer in self.layers[:-1]:
                        inputs = F.relu(layer(inputs))
                inputs = layers[-1](inputs)

                return  inputs
