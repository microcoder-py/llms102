import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange

from feedforward import FFN 
from layer_norm import RMSNorm

"""
Special thanks to @JannisZeller for his very well written implementation of RoPE 
https://github.com/JannisZeller/rope-multi-head-attention/blob/main/src/rope_attention_layer.py
"""
class MHAWithRoPE(nn.Module):
    def __init__(self, emb_dim, num_heads, max_seq_len):
        super().__init__()

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        proj_dim = emb_dim // num_heads
        self.register_buffer("proj_dim", torch.tensor(proj_dim))
        self.register_buffer("num_heads", torch.tensor(num_heads))
        self.register_buffer("scale", torch.tensor(proj_dim ** -0.5))

        #Caching the sine and cosine values by position to apply
        #rotary embeddings
        cosines, sines = self._get_rotary_embedding_matrix(emb_dim, max_seq_len)
        self.register_buffer('cosines', cosines)
        self.register_buffer('sines', sines)

        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        self.w_o = nn.Linear(emb_dim, emb_dim)

    def _get_rotary_embedding_matrix(self, dim, max_seq_len = 128000):
        half_dim = dim // 2
        thetas = 1.0 / (10000 ** (torch.arange(half_dim) / half_dim))

        positions = torch.arange(max_seq_len)

        pos_thetas = torch.outer(positions, thetas)

        cosines = torch.cos(pos_thetas)
        sines = torch.sin(pos_thetas)

        cosines = torch.repeat_interleave(cosines, 2, dim = -1)
        sines = torch.repeat_interleave(sines, 2, dim = -1)

        return cosines, sines
    
    def _order_for_rotary_embedding(self, x):
        #Cosine dot product needs all x values in order
        #Sine dot product needs pairwise inversion, followed by
        #negation of first term. 
        #Refer http://llms102.streamlit.app/positional_encoding#the-rotation-in-2-d
        x_odd = x[..., ::2]
        x_even = x[..., 1::2]
        x_stacked = torch.stack([-x_even, x_odd], dim = -1)
        return x_stacked.flatten(start_dim=-2)
    
    def _apply_rotary_embedding(self, x):
        seq_len = x.shape[-2]
        x_for_sine = self._order_for_rotary_embedding(x)

        cos = self.cosines[:seq_len, :self.proj_dim] 
        sin = self.sines[:seq_len, :self.proj_dim]

        x_rope = x * cos + x_for_sine * sin
        return x_rope
    
    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = torch.reshape(q, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)
        k = torch.reshape(k, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)
        v = torch.reshape(v, (batch_size, seq_len, self.num_heads, self.proj_dim)).transpose(2, 1)

        q_rope = self._apply_rotary_embedding(q)
        k_rope = self._apply_rotary_embedding(k)

        attention_scores = q_rope @ k_rope.transpose(-2, -1)
        scaled = attention_scores * self.scale
        softmaxed = F.softmax(scaled, dim = -1)

        scaled_values = softmaxed @ v

        scaled_values = rearrange(scaled_values, 'b n s p -> b s (n p)', n = self.num_heads)

        output = self.w_o(scaled_values)

        return output

class TransformerBlock(nn.Module): 
    def __init__(self, hidden_dim, num_attn_heads, max_seq_len, epsilon):
        super().__init__()

        self.attention = MHAWithRoPE(emb_dim = hidden_dim, num_heads = num_attn_heads, max_seq_len = max_seq_len)
        self.ffn = FFN(embed_dim = hidden_dim, up_proj_multiple = 4)
        self.layerNorm1 = RMSNorm(input_dim = hidden_dim, epsilon = epsilon)
        self.layerNorm2 = RMSNorm(input_dim = hidden_dim, epsilon = epsilon)
        
    def forward(self, x):
        """
        Shortest code would be 

        x = x + self.attention(self.layerNorm1(x))
        x = x + self.ffn(self.layerNorm2(x))
        return x
        """
        x_unnorm = x
        #Attn Pre-norm
        x = self.layerNorm1(x)
        
        #Attention
        x = self.attention(x)

        #Residual Add
        x = x + x_unnorm 

        x_unnorm2 = x

        #FFN Pre-norm
        x = self.layerNorm2(x)
         
        #FFN
        x = self.ffn(x)

        #Residual Add
        x = x + x_unnorm2 

        return x
    
class TransformerNetwork(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            max_seq_len, 
            norm_eps, 
            vocab_size, 
            num_blocks,
            num_attn_heads,
            ignore_index
        ):

        super().__init__()

        self.ignore_index = ignore_index

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim,
            padding_idx = -1
        )

        transformer_blocks = [
            TransformerBlock(
                hidden_dim = embed_dim, 
                num_attn_heads = num_attn_heads, 
                max_seq_len = max_seq_len, 
                epsilon = norm_eps
            )

            for i in range(num_blocks)   
        ]

        self.transformer_blocks = nn.ModuleList(transformer_blocks) 

        self.preLMHeadNorm = RMSNorm(input_dim = embed_dim, epsilon = norm_eps) 

        self.LMHead = nn.Linear(embed_dim, vocab_size, bias = False)
        
        #Weight tying the LM Head and Embedding matrix
        self.LMHead.weight = self.embedding.weight

        assert self.LMHead.weight.data_ptr() == self.embedding.weight.data_ptr(), "LMHead weight tying failed, please verify code"

    def forward(self, x, targets = None):
        #x is a sequence of token indices 
        #Need to generate embeddings
        x = self.embedding(x) 

        """
        If we had used Sinusoidal positional encoding, here is where we would add it 

        x = x + pos_enc(x)
        """

        for block in self.transformer_blocks:
            x = block(x) 

        x = self.preLMHeadNorm(x)

        logits = self.LMHead(x)

        if targets is not None: 
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
            return logits, loss

        return logits