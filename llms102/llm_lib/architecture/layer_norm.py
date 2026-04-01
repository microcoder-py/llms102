import torch.nn as nn 
import torch 

#Official Implementation: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/normalization.py#L106
class LayerNorm(nn.Module):
    def __init__(self, input_dim, epsilon):
        super().__init__()

        #Init with ones to provide no scaling per dim initially
        self.gamma = nn.Parameter(torch.ones(input_dim))

        #Init with zeros to provide no bias initially
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, x): 
        per_token_mean = torch.mean(x, dim = -1, keepdim=True) 

        #Official pytoch implementation uses 
        #sqrt(sum((x - mean)^2 + eps))
        per_token_std = torch.sqrt(torch.mean((x - per_token_mean)**2 + self.epsilon, dim = -1, keepdim=True))

        centered = x - per_token_mean
      
        normalized = centered/per_token_std

        output = normalized * self.gamma + self.bias 

        return output

#Official Pytorch:
#https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/nn/modules/normalization.py#L335
class RMSNorm(nn.Module):
    def __init__(self, input_dim, epsilon):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, x):
        normalization_factor = torch.sum(torch.square(x), dim = -1, keepdim = True)/x.shape[-1] + self.epsilon
        normalization_factor = torch.sqrt(normalization_factor)

        rms_norm = self.gamma * x/normalization_factor 

        return rms_norm