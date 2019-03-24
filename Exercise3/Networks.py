import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
        def __init__(self, features_level="high"):
                super(ValueNetwork, self).__init__()
                hidden_sizes = [50, 4]
                if features_level == "high":
                        input_dim = 15
                else:
                        input_dim = 68

                self.layers = nn.ModuleList()

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                for size in hidden_sizes:
                        self.layers.append(nn.Linear(input_dim, size))
                        input_dim = size
                
        def forward(self, inputs) :
                #inputs = inputs.float()
                inputs = torch.tensor(inputs).to(self.device)
                for layer in self.layers[:-1]:
                        inputs = F.relu(layer(inputs))
                inputs = self.layers[-1](inputs)
                #inputs = torch.nn.Softmax(inputs) # We are not calculating probabilities

                return inputs
