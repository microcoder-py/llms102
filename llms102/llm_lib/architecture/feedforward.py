import torch.nn as nn 
import torch 

class FFN(nn.Module):
    def __init__(self, embed_dim, up_proj_multiple):
        super().__init__()

        self.up_proj = nn.Linear(
            in_features = embed_dim, 
            out_features = embed_dim * up_proj_multiple,
            bias = True    
        )

        #We can use other activations as well
        self.activation = nn.SiLU()

        self.down_proj = nn.Linear(
            in_features = embed_dim  * up_proj_multiple, 
            out_features = embed_dim,
            bias = True    
        )

    def forward(self, x):
        output = self.down_proj(self.activation(self.up_proj(x))) 
        return output