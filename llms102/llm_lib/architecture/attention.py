import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, embedding_dims, proj_dim):
      super().__init__()

      #Typically you'd use a linear layer to include bias, 
      #omitting for code clarity. 
      self.w_q = nn.Parameter(torch.randn(embedding_dims, proj_dim))
      self.w_k = nn.Parameter(torch.randn(embedding_dims, proj_dim))
      self.w_v = nn.Parameter(torch.randn(embedding_dims, proj_dim))

      norm_factor = torch.sqrt(torch.tensor(proj_dim))
      self.register_buffer('norm_factor', norm_factor)

    def forward(self, token_batch):
      q_proj = token_batch @ self.w_q
      k_proj = token_batch @ self.w_k
      v_proj = token_batch @ self.w_v

      attention_scores = q_proj @ k_proj.transpose(-2, -1) / self.norm_factor

      attention_softmaxed = nn.functional.softmax(attention_scores, dim = -1)

      output = attention_softmaxed @ v_proj

      return output