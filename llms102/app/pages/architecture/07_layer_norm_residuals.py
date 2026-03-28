# app/pages/architecture/07_layer_norm_residuals.py

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from graphviz import Digraph
from utils.code_loader import show_source, run_it_yourself

st.set_page_config(page_title="Residuals & Layer Norm | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Residuals & Layer Norm")
st.caption("Making sure nobody screams, and everybody learns")

st.markdown("""
We've all engaged in gossip whether we like it or not. A common hollywood trope is the idea that a piece of gossip finds itself imbued with the creativity of the speaker and listener, ultimately taking on a life of its own. For clarity - no, I did not mean to start a lamb racing arena in sixth grade. I had suggested lambs and racing in separate contexts, but then the gossip chains began. 
            
Transformers suffer from a similar problem - as we go deeper into the layers, the activations and gradients start taking on a life of their own. We need to introduce something to stabilize them, ensuring no activations explode, and to ensure that gradients have a clear path to flow through. For this, we introduce Residuals, and Layer Norms
""")  # your intro — pull from your newsletter, 2-3 sentences

st.divider()

# ─── 1. The vanishing gradient problem ───────────────────────────────────────

st.header("The vanishing gradient problem")

st.markdown("""
For the network to be able to grasp several different levels of information, we need more depth. This is also the reason why we now have trillion parameter models - they all focus on a lot of depth. 

However, the deeper that we go, the information being passed back to shallower layers gets smaller and smaller during backprop. This introduces a strange dichotomy - the initial layers provide a lot of early semantic information without which later layers would struggle, but these are the layers that suffer most from vanishing gradients.

As a simple example, let's assume that gradient magnitudes drop off geometrically with layer depth and explore how the information passed back declines.  
""")  

st.markdown("**Gradient degradation across layers — interactive**")

num_layers = st.slider("Number of layers", min_value=2, max_value=32, value=12)
gradient_factor = st.slider("Gradient scaling factor per layer", min_value=0.1, max_value=0.99, value=0.7, step=0.01)

layers = np.arange(num_layers)
gradients = gradient_factor ** layers

fig = px.line(
    x=layers,
    y=gradients,
    labels={"x": "Layer (from output)", "y": "Gradient norm"},
    title=f"Gradient norm across {num_layers} layers (factor={gradient_factor})"
)
fig.update_traces(line=dict(color="#FF6B6B", width=2))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"With a gradient scaling factor of {gradient_factor} per layer, "
    f"the gradient at layer 0 is {gradient_factor**num_layers:.6f} — "
    f"{'effectively zero' if gradient_factor**num_layers < 0.01 else 'still meaningful'}. "
    f"Early layers receive almost no learning signal."
)

st.divider()

# ─── 2. Residual connections ─────────────────────────────────────────────────

st.header("Residual connections")

st.markdown("""
The original idea for residual connections comes from a 2015 Microsoft research paper He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)where they realised that a simple fix for the vanishing gradient problem comes from providing some residual connections. Ideally, deeper networks should have more representation capacity, so at the very least, we should be seeing equal or lower losses. However, they found the opposite - losses actually increased with depth on their task.
            
Their solution was to provide **Residual Connections**, a pathway through which the original information was better preserved. During backpropagation, this provides a sort of 'gradient highway', i.e. an alternative path through which gradients can flow, assisting with further stabilization of the network.

In the graph below, notice the red lines - those are the residual connections.  
""")  # pull from newsletter
                 # answer: the simplest possible fix — add input back to output
                 #         gradient highway — backprop flows through addition directly
                 #         why addition not multiplication — additive identity
                 #         He et al. 2015 — where residuals came from

def build_residual_graph():
    dot = "digraph Residual {\n"
    dot += 'rankdir=TB;\n'
    dot += 'graph [pad="0.5", nodesep="0.6", ranksep="1.2"];\n'
    dot += 'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=24 width=4];\n'

    # Nodes (same vertical line)
    dot += f'X   [label="Input", fillcolor="#E3F2FD"];\n'
    #dot += f'LN1 [label="LayerNorm", fillcolor="#E8F5E9"];\n'
    dot += f'ATTN [label="Multi-Head Attention", fillcolor="#FFF3E0"];\n'
    dot += f'ADD1 [label="Add (Residual)", fillcolor="#FFEBEE"];\n'

    #dot += f'LN2 [label="LayerNorm", fillcolor="#E8F5E9"];\n'
    dot += f'FFN1 [label="Linear", fillcolor="#FFF3E0"];\n'
    dot += f'ACT  [label="Activation", fillcolor="#E8F5E9"];\n'
    dot += f'FFN2 [label="Linear", fillcolor="#FFF3E0"];\n'
    dot += f'ADD2 [label="Add (Residual)", fillcolor="#FFEBEE"];\n'

    dot += f'OUT [label="Output", fillcolor="#E1F5FE"];\n'

    # Main vertical flow
    dot += "X -> ATTN -> ADD1 -> FFN1 -> ACT -> FFN2 -> ADD2 -> OUT;\n"

    # Residual connections (side arrows)
    dot += 'X -> ADD1 [color="#E74C3C", penwidth=3];\n'
    dot += 'ADD1 -> ADD2 [color="#E74C3C", penwidth=3];\n'

    dot += "}\n"
    return dot


# ---- Streamlit layout fixes ----

# Center the graph
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.graphviz_chart(
        build_residual_graph(),
        use_container_width=True
    )

st.markdown(r"""
First, let's write out the equation for what would happen if we used residual connections. We basically take the output of a given layer, and add the initial output to it

$$
\text{output} = x + \text{sublayer}(x)
$$
            
Now let's see how the gradient for this looks 
$$            
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(
I + \frac{\partial F(x)}{\partial x}
\right)
$$

Notice how there is a clear separation of concerns. Even if the gradient flowing through the sublayer (i.e. second term inside brackets) is vanishing, we still have an identity pathway that allows the gradient information to flow through. This equation does not introduce any new parameters - just a simple addition allows us to provide a path for the gradient to flow, ensuring all layers learn.

The additive residual is also by design. It provides a simple identity pass through for gradients. I did try and explore if there were other variants, such as multiplicative or concatenated, but haven't yet been able to find significant information - it most likely has to do with the stability of the identity pass through. With multiplicative, we would once again see the vanishing gradient problem arise. I personally don't think we need to read too much into it, but the interested reader is as always invited to explore. There are some implementations of multiplicative residual connections in computer vision, physical process modelling and others, where the multiplicative connection is combined with some form of non-linearity to expressly capture more complex interactions, but not much for language modelling. 
            
Intuitively, it also means that the inputs are not being remodelled entirely - they are being 'updated' since we do pass the original input back to the output. The model likely will then learn how to update, not entirely change the inputs. 
""")

st.header("Layer normalisation")

st.markdown("""
We've seen how to make sure a learning signal is reaching all the layers by using residual connections. However, we haven't yet tempered the learning signal itself. If the activations themselves start to explode, gradients will deteriorate, and the learning signal being passed back will be poor in quality. 

            
""")  # study: Ba et al. 2016 — Layer Normalization arxiv 1607.06450
                 # answer: what happens without normalisation — activations explode
                 #         layer norm normalises each token independently
                 #         mean 0 variance 1 across the embedding dimension
                 #         learnable scale and shift — gamma and beta

st.latex(r"\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta")

st.markdown("")  # answer: what gamma and beta do,
                 #         why epsilon — numerical stability,
                 #         difference from batch norm — why batch norm fails for sequences

st.markdown("Ba et al. (2016) — [Layer Normalization](https://arxiv.org/abs/1607.06450)")

# ── layer norm widget ─────────────────────────────────────────────────────────

st.subheader("Layer normalisation — interactive")

st.markdown("")  # one sentence directing the reader

np.random.seed(42)
raw = np.random.randn(16) * 5 + 3

mean = raw.mean()
std = raw.std()
normalised = (raw - mean) / (std + 1e-5)

col1, col2 = st.columns(2)

fig1 = px.bar(
    x=list(range(16)), y=raw,
    title=f"Before LayerNorm (mean={mean:.2f}, std={std:.2f})",
    labels={"x": "Dimension", "y": "Value"},
    color=raw, color_continuous_scale="RdBu"
)
fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    x=list(range(16)), y=normalised,
    title=f"After LayerNorm (mean≈0, std≈1)",
    labels={"x": "Dimension", "y": "Value"},
    color=normalised, color_continuous_scale="RdBu"
)
fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
col2.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── 4. Pre-norm vs post-norm ────────────────────────────────────────────────

st.header("4. Pre-norm vs post-norm")

st.markdown("")  # study: reason from first principles + Xiong et al. 2020 arxiv 2002.04745
                 # answer: original Vaswani paper used post-norm
                 #         modern transformers use pre-norm — more stable
                 #         pre-norm: x + sublayer(LayerNorm(x))
                 #         post-norm: LayerNorm(x + sublayer(x))

col1, col2 = st.columns(2)

col1.markdown("**Post-norm (original)**")
col1.latex(r"x = \text{LayerNorm}(x + \text{sublayer}(x))")
col1.markdown("")  # one sentence — why this is less stable

col2.markdown("**Pre-norm (modern standard)**")
col2.latex(r"x = x + \text{sublayer}(\text{LayerNorm}(x))")
col2.markdown("")  # one sentence — why this trains better

st.markdown("Xiong et al. (2020) — [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)")

st.divider()

# ─── 5. Putting it together ───────────────────────────────────────────────────

st.header("5. Putting it together")

st.markdown("")  # answer: the full pre-norm transformer sublayer
                 #         one block = attention + FFN, each wrapped in residual + norm
                 #         residuals carry the signal, norm stabilises it

st.latex(r"""
\begin{aligned}
x &= x + \text{Attention}(\text{LayerNorm}(x)) \\
x &= x + \text{FFN}(\text{LayerNorm}(x))
\end{aligned}
""")

st.divider()

# ─── 6. Implementation ───────────────────────────────────────────────────────

st.header("6. Implementation")

st.markdown("")  # one sentence

show_source("llm_lib/architecture/layer_norm.py")

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

run_it_yourself("""
from llm_lib.architecture.layer_norm import PreNormResidual
import torch

layer = PreNormResidual(d_model=512)
x = torch.randn(1, 10, 512)
output = layer(x)
print(output.shape)   # (1, 10, 512)
""")

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/08_positional_encoding.py", label="Next: Positional Encoding →")

st.markdown("""
**Further reading**
- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Ba et al. (2016) — [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Xiong et al. (2020) — [On Layer Normalization in the Transformer Architecture](https://arxiv.0rg/abs/2002.04745)
""")