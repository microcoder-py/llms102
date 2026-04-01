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
We've seen how to make sure a learning signal is reaching all the layers by using residual connections. However, we haven't yet tempered the learning signal itself. If the activations themselves start to explode/vanish, gradients will deteriorate, and the learning signal being passed back will be poor in quality. 

There is a problem formally referred to as **internal covariate shift**, which basically means that when layers interact with each other, after processing, there is significant shift in the data distribution. This theory was systematically studied in Ioffe et. al, 2015 - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167), one of the first succesful implementations of normalization of any kind. 

It was established previously that network training converges faster if inputs are 'whitened' - i.e. linearly transformed to have zero means and unit variances. The authors then assert that in deeper networks, each layer provides input to the next. Therefore, would whitening the inputs at each step help in stabilization? In particular, the authors suggest that the normalization step needs to be a part of the gradient flow as well, so that the layer adapts and helps the other layers adapt to these changes to inputs. The empirical answer was yes, although later ablations on BatchNorm seem to provide a different theoretical basis. Regardless, it's clear that normalization is beneficial while training deep networks. 

We don't use batchnorm in LLMs for the most part. We use LayerNorm (or its variants). LayerNorm performs this normalization per-token as opposed to per batch or sequence. The original LayerNorm paper provides plenty of theoretical basis for why it works, including the deeper geometric analysis, and is recommended. Let's take a look at the formula
""")  
st.latex(r"\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta")

st.markdown(r"""
Here, $\mu$ is the mean of $x$, $\sigma$ is the standard deviation, and $\gamma$ is a learned parameter, while $\beta$ is the bias. Let's develop an intuition for what we are seeing here
            
1. First, we recenter the distribution around 0. This is the subtraction from mean term in the numerator 
2. Next, we normalize the variance of the distribution dividing by the standard deviation (plus some small constant for numerical stability). Remember that $\sigma^2(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$. Therefore, the variance of the normalized values becomes 1. The reader is invited to derive this by themselves using the formulae provided. $\epsilon$ is added for numerical stability in case the variance of the input distribution is too small, causing very large division
3. Now, we have normalized the values. But what if the layer actually needed rescaling to converge better? What if by forcing all values to be bounded with our linear transform, we have reduced the expressivity of the network? Therefore, we add a gain parameter, $\gamma$. This is a vector of the same dimensions as $x$, providing a per-dimension scaling factor, learned during training 
4. Bias is added as a secondary learned parameter
""")  
st.markdown("> Ba et al. (2016) — [Layer Normalization](https://arxiv.org/abs/1607.06450)")

# ── layer norm widget ─────────────────────────────────────────────────────────

st.markdown("**Layer normalisation — interactive**")

st.markdown("Below, you can see the effects of LayerNorm on random inputs. The distribution is much smoother, and passing this through ReLU or other non-linearities should result in lesser neurons dying. Additionally, gradients will flow more smoothly (since we need smaller updates per input), resulting in lesser variance of output and gradient values.")  # one sentence directing the reader
np.random.seed(42)
raw = np.random.randn(16) * 5 + 3

mean = raw.mean()
std = raw.std()
normalised = (raw - mean) / (std + 1e-5)

# Compute shared symmetric y-axis range
abs_max = max(np.abs(raw).max(), np.abs(normalised).max())
y_range = [-abs_max, abs_max]

# Layout
col1, col2 = st.columns(2)

# Before LayerNorm
fig1 = px.bar(
    x=list(range(16)), y=raw,
    title=f"Before LayerNorm (mean={mean:.2f}, std={std:.2f})",
    labels={"x": "Dimension", "y": "Value"},
    color=raw, color_continuous_scale="RdBu"
)
fig1.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
fig1.update_yaxes(range=y_range)
col1.plotly_chart(fig1, use_container_width=True)

# After LayerNorm
fig2 = px.bar(
    x=list(range(16)), y=normalised,
    title="After LayerNorm (mean≈0, std≈1)",
    labels={"x": "Dimension", "y": "Value"},
    color=normalised, color_continuous_scale="RdBu"
)
fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
fig2.update_yaxes(range=y_range)
col2.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("RMSNorm")

st.markdown(r"""
In LayerNorm, we first recentered the data, and then normalized it. In their paper, [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467) the authors found that most of these benefits can be achieved without the recentering, and simply rescaling. This turns out to be faster and their formulation also doesn't need to calculate the variance of the input, providing further speedup. The formula is 
            
$$
\mathrm{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}}          
$$
            
Newer models such as Llama and Mistral series employ RMSNorm instead of LayerNorm
""")

# ─── 4. Pre-norm vs post-norm ────────────────────────────────────────────────

st.header("Where does this come within the transformer network?")

st.markdown("""
Normalization is provided somewhere around the attention and FFN networks, the main compute units within the transformer blocks. The main question then becomes, _when_ do we apply normalization? 
            
Broadly speaking, there are two variants, **postnorm** and **prenorm**. 
            
In Post-Norm, we normalize the outputs **after** passing it through the relevant layer (MHA/FFN). Original transformers paper, BERT, etc. used this paradigm.
            
In Pre-Norm, we we normalize the inputs **incoming** to the relevant layer (MHA/FFN). Empirically, this has shown to provide better stability. Theoretically, recall our gradient flow equation from before. With Pre-Norm, the identity path for gradient flow occurs along the skip path directly, preserving more of the gradient signal for the input since it did not undergo rescaling or normalization. GPT-2 was the first popular model to use Pre-Norm, and it became the defacto standard for many models. Below are graphical representations of both.
""")  # study: reason from first principles + Xiong et al. 2020 arxiv 2002.04745
                 # answer: original Vaswani paper used post-norm
                 #         modern transformers use pre-norm — more stable
                 #         pre-norm: x + sublayer(LayerNorm(x))
                 #         post-norm: LayerNorm(x + sublayer(x))

def build_norm_graph(mode = "pre"):
    dot = "digraph Residual {\n"
    dot += 'rankdir=TB;\n'
    dot += 'graph [pad="0.5", nodesep="0.6", ranksep="1.2"];\n'
    dot += 'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=24 width=4];\n'

    # Nodes (same vertical line)
    dot += f'X   [label="Input", fillcolor="#E3F2FD"];\n'
    dot += f'LN1 [label="LayerNorm", fillcolor="#E8F5E9"];\n'
    dot += f'ATTN [label="Multi-Head Attention", fillcolor="#FFF3E0"];\n'
    dot += f'ADD1 [label="Add (Residual)", fillcolor="#FFEBEE"];\n'

    dot += f'LN2 [label="LayerNorm", fillcolor="#E8F5E9"];\n'
    dot += f'FFN1 [label="FFN", fillcolor="#FFF3E0"];\n'
    dot += f'ADD2 [label="Add (Residual)", fillcolor="#FFEBEE"];\n'
    
    dot += f'OUT [label="Output", fillcolor="#E1F5FE"];\n'

    # Main vertical flow
    if mode == "post":
        dot += "X -> ATTN -> ADD1 -> LN1 -> FFN1 -> ADD2 -> LN2 -> OUT;\n"
    else:
        dot += "X -> LN1 -> ATTN -> ADD1  -> FFN1 -> LN2 ->  ADD2 -> OUT;\n"
    # Residual connections (side arrows)
    dot += 'X -> ADD1 [color="#E74C3C", penwidth=3];\n'
    dot += 'ADD1 -> ADD2 [color="#E74C3C", penwidth=3];\n'

    dot += "}\n"
    return dot


# ---- Streamlit layout fixes ----
col1, col2 = st.columns(2)

col1.markdown("**Post-norm (original)**")
col1.latex(r"x = \text{LayerNorm}(x + \text{sublayer}(x))")

col2.markdown("**Pre-norm (modern standard)**")
col2.latex(r"x = x + \text{sublayer}(\text{LayerNorm}(x))")

coln1, coln2 = st.columns([3, 3])


with coln1:
    st.graphviz_chart(
        build_norm_graph(mode = "post"),
        use_container_width=True
    )

with coln2:
    st.graphviz_chart(
        build_norm_graph(mode = "pre"),
        use_container_width=True
    )

st.markdown("""
However, these are not the only paradigms. For instance, Gemma 3 uses Pre+Post Norm. Some models use QK-Norm (i.e. normalizing the QK projections). Some models such as Olmo2 are going back to Post Norm with QK Norm, including decisions regarding when to re-add positional encoding (to be discussed in next chapter). These are architecture specific details but worth highlighting for those who may want to explore it further. The intuition regarding why we use normalization holds regardless.
""")

st.divider()

# ─── 6. Implementation ───────────────────────────────────────────────────────

st.header("Implementation")

st.markdown("Simple implementation of the formula")  # one sentence

show_source("llms102/llm_lib/architecture/layer_norm.py")

st.divider()

col1, col2 = st.columns([3, 5])
col1.page_link("pages/architecture/08_positional_encoding.py", label="Next: Positional Encoding →")
