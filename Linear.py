import numpy as np
import random
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self,input_dim, output_dim,weights,bias):
        super().__init__()
        self.Linear = nn.Linear(input_dim,output_dim)
        with torch.no_grad():
            self.Linear.weight = torch.nn.Parameter(weights)
            self.Linear.bias = torch.nn.Parameter(bias)

    def Forward(self,input):
        normalized_input = self.normalizeInput(input)
        output = self.Linear(normalized_input)
        return output

    def normalizeInput(self,input):
        col_norms = torch.norm(input, p=2, dim=0, keepdim=True)
        normalized_x = input / col_norms  # Normalize columns
        return normalized_x