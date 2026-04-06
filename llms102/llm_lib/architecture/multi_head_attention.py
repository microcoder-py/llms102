import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__() 

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        
        #Instead of us creating specific proj dims, 
        #official pytorch implementation divides the emb_dim
        #by num_heads to find proj_dim
        proj_dim = emb_dim // num_heads
        self.register_buffer("proj_dim", torch.tensor(proj_dim))
        self.register_buffer("num_heads", torch.tensor(num_heads))
        self.register_buffer("scale", torch.tensor(proj_dim ** -0.5))

        #We use a large matrix, perform the projection, then 
        #split and reshape it to give us individual heads
        #Larger matrices allow for faster compute on GPUs
        #if saturated appropriately. Smaller matrices means
        #repeat fetches which is slower
        #Official implementation goes further, storing a 
        #single matrix with all three stacked which is then 
        #split into QKV matrices at runtime
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        self.w_o = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape 
        
        #Calculate QKV projs for input
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x) 

        #Split into individual heads. We can use einops as well
        q = torch.reshape(q, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)
        k = torch.reshape(k, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)
        v = torch.reshape(v, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)
        
        #Attention Formula Application
        #Because we split the heads, even though 
        #it looks like one large matmul, we have 
        #separate attention scores computed for each head
        attention_scores = q @ k.transpose(-2, -1) 
        scaled = attention_scores * self.scale 
        softmaxed = F.softmax(scaled, dim = -1) 

        scaled_values = softmaxed @ v 

        #Need to reshape into emb_dim, emb_dim to matmul with w_o
        scaled_values = rearrange(scaled_values, 'b n s p -> b s (n p)', n = self.num_heads) 

        output = self.w_o(scaled_values)

        return output
