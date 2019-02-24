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
                hidden_sizes = [50, 1]
                input_dim = 68+1
                self.layers = []
                self.layers = nn.ModuleList()

                for size in hidden_sizes:
                        self.layers.append(nn.Linear(input_dim, size))
                        input_dim = size
                
        def forward(self, inputs) :
                inputs = inputs.float()
                for layer in self.layers[:-1]:
                        inputs = F.relu(layer(inputs))
                inputs = self.layers[-1](inputs)

                return inputs
