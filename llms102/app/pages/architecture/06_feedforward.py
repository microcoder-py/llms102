# app/pages/architecture/06_feedforward.py

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.code_loader import show_source, run_it_yourself

st.set_page_config(page_title="Feedforward Networks | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Feedforward Networks")
st.caption("llms102 — the last lesson you will need")

# ─── 1. What does the FFN do ─────────────────────────────────────────────────

st.header("What does the FFN do?")

st.markdown("""
Now that we have applied attention, we have intermixed tokens, and found out which one relies on which and by how much. However, we still haven't performed any processing on the intermixed tokens. 
            
The module that allows processing once tokens are attended to, is the feedforward network. The intuition for it is provided in a later section, for now, follow along with the formulation first. 
""")  

st.divider()

# ─── 2. The formula ──────────────────────────────────────────────────────────

st.header("The formula")

st.markdown("""
The output from the multihead attention layer has the same dimensions as the original sequence, i.e. `(batch, seq_len, embed_dim)` 

The feedforward-network first projects this into a higher dimensional space (4x size in the original implementation, may vary), followed by a nonlinearity (ReLU in original paper). This is then again projected down into the original size space by a different learned matrix. Overall, the equation is:             
""")
st.latex(r"\text{FFN}(x) =  f(W_1 x + b_1) \cdot W_2  + b_2")

st.markdown("""
The original paper does not provide any information about what the FFN actually does, except for a short passage. Further research conducted ex post facto on interpretability offers better explanations. For now, it is sufficient to understand that the FFN layer stores information about the tokens in a higher dimensional space for more expressivity. The attached non-linearity allows more complex transformations. 

Remember - attention focuses on interactions BETWEEN tokens. FFN focuses on information content OF THE TOKEN, allowing the network to provide richer representation and refinement across steps. 
            
One other point to make regarding processing efficiency. As you would remember, in MHA, we had separate heads that operated independently, which let us parallelize their computation. Similarly, since in the FFN tokens are processed independently of each other, we can splice the corresponding matrices without having to worry about inter token interactions, allowing more efficient compute. 
""")  

st.divider()

# ─── 3. Activation functions ─────────────────────────────────────────────────

st.header("Activation functions")

st.markdown("""
In the attention layer, we only had linear interactions. If we trained a complete neural network with just linear transformations, the extended network would still be a linear one (composition of linear transforms), and would not be able to model complex, non-linear and non monotonic patterns. The FFN is the part of the transformer architecture that allows for this. 
            
It is important to understand what activation functions are used. Without diving too deep, the original architecture used ReLU, but today we use other activations such as GeLU and SwiGLU. The main issue with ReLU was that it has a tendency to cause the 'dead neuron' problem (i.e. too many 0 output activations), which enforced excess sparsity and therefore killed a lot of gradient information. GeLU, SiLU and SwiGLU provide smoother, differentiable transitions, allowing more information to flow through. The analysis and benefits of each nonlinearity are left to the reader's perusal. 
""")  

x = np.linspace(-4, 4, 300)

relu = np.maximum(0, x)
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
silu = x * (1 / (1 + np.exp(-x)))  # SiLU / Swish — used in SwiGLU

activations = st.multiselect(
    "Select activations",
    options=["ReLU", "GELU", "SiLU"],
    default=["ReLU", "GELU", "SiLU"]
)

fig = go.Figure()
mapping = {"ReLU": (relu, "#FF6B6B"), "GELU": (gelu, "#4ECDC4"), "SiLU": (silu, "#F7DC6F")}

for name in activations:
    vals, color = mapping[name]
    fig.add_trace(go.Scatter(x=x, y=vals, name=name, line=dict(color=color, width=2)))

fig.update_layout(
    title="Activation functions",
    xaxis_title="x",
    yaxis_title="activation(x)",
    legend=dict(orientation="h"),
    height=400
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── 5. FFN as memory ────────────────────────────────────────────────────────

st.header("FFN as memory")

st.markdown("> Geva et al. (2021) — [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)")

st.markdown("""
When scientists say that nobody truly knows why transformers understand what they understand, they aren't joking. The reality of this field is that a lot of work is done empirically, of course with a lot of highly informed intuition behind what could work, but when it _does_ work, it's incredibly hard to explain why. 
            
There is an entire active field of research dedicated to this pursuit - it is called mechanistic interpretability, which conducts controlled experiments on neural networks to break down characteristics of why each layer works and why. 
            
In the paper we tagged above, the scientists sought to analyze what the FFN truly represents. They assert that the two layers function in effect as unnormalized Key-Value memories. Notice our above activation function set: at values below zero, all of them greatly shrink the activation magnitude close to 0. This is the (approx.) positive gate, keeping track of what information needs to be refined or not. The second matrix acts as a Values store, retaining information about the token (i.e. acting as a distribution over vocabulary), and based on how much it is activated using the Key store, we get specific information extracted. This provides strong evidence that the FFN network plays a key role in language modelling, directly impacting how the model interprets the distribution.
            
Through experiments, they find that **the key-value memories are highly associated with human recognizable patterns**. They also found evidence that shallow layers detect shallow patterns while deeper layers focused more on semantic information and correlation. Each layer processes and parses information at different levels - the combination of attention (look at this token for relevant information) along with the FFN (this token has this specific nuance, capture it) are what allow transformers to be so effective. 
            
Now, to be clear, this is a field with ongoing work. The discussion of one single paper does not provide definitive evidence as to what they do, but it does provide considerable intuition. The interested reader may look up interpretability as a subject. It is honestly a very very vast field, and cannot be covered under this tutorial. 
""") 

st.divider()

# ─── 6. Implementation ───────────────────────────────────────────────────────

st.header("Implementation")

st.markdown("Fairly straightforward, just receiving an input, up-projecting to higher dimension, applying some non-linearity to it, and then down-projecting it back to original dimension") 

show_source("llms102/llm_lib/architecture/feedforward.py")

st.divider()

run_it_yourself("""
from llm_lib.architecture.feedforward import FFN
import torch

ffn = FFN(embed_dim=512, up_proj_multiple=4)
x = torch.randn(1, 10, 512)  # (batch, seq_len, d_model)
output = ffn(x)
print(output.shape)           # (1, 10, 512)
""")

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/07_layer_norm_residuals.py", label="Next: Layer Norm & Residuals →")
