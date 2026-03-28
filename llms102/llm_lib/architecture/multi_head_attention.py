from attention import SelfAttention 

import torch.nn as nn 
import torch 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()  

        if embed_dim % num_heads != 0: 
            raise ValueError("Embedding dimension must be a multiple of num_heads")

        proj_dim = int(embed_dim/num_heads)
      
        self.heads = nn.ModuleList(
            [SelfAttention(embedding_dims=embed_dim, proj_dim=proj_dim) 
            for _ in range(num_heads)] 
        )

        self.w_o = nn.Parameter(torch.randn(num_heads * proj_dim, embed_dim))


    def forward(self, x):
        #Inefficient implementation, runs each head sequentially
        #but enough to explain the concept
        head_ops = torch.cat([head(x) for head in self.heads], dim = -1)
        o_proj = head_ops @ self.w_o 

        return o_proj