# app/pages/architecture/08_positional_encoding.py

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Positional Encoding | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Positional Encoding")
st.caption("Teaching transformers to care about order")

st.markdown("""
Anyone with a basic grasp of English knows that 'dog bites man' and 'man bites dog' are very different statements, one a commonality, the other raising shock and awe. To a transformer though, they look the same. So far, we have assumed that we are using a sequence, and this position invariance is obvious to us. However, in the attention mechanism in particular, the model effectively has no positional information. Even if we changed the word order, attention would work as usual and the rest of the model would compensate for the loss incurred, degrading performance. 
            
Positional information also informs the model of a sense of locality - closer words are more likely to be more highly correlated compared to words far away. That said, as we venture into multi-million token context windows, we also need to effectively capture the semantics of tokens far away from each other. 
            
Positional encodings help us solve this. 
""")
# TODO: Intro anecdote — something about how meaning changes with order
# (e.g. "dog bites man" vs "man bites dog", or a recipe where order matters)
# ~3 sentences, conversational tone, mirrors the gossip framing in layer norm
st.divider()

# ─── 1. Why position matters ─────────────────────────────────────────────────

st.header("Why does position matter?")

st.markdown(r"""
Attention is a permutation invariant operation. It does not care about what the relative ordering of the tokens is, in particular the $QK$ matrix. The attention mechanism accepts a set, not a sequence. Language, however, is not permutation invariant. 
            
This mechanism was created to eliminate the weaknesses of recurrent networks and convolution, but both these alternatives provide an inherent sequential processing which is missing in attention at present. 

The original transformers paper only provides three paragraphs of information, but the one sentence that carries most weight is: 
            
> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$
            
A lot of the theory behind PE was created ex post facto. The original paper only references one other paperr that works on it, and neither provide any detailed explanations. Similar to the case of FFNs, I surmise that this was based on informed intuition, which was later analyzed in depth to separate concerns between different parts of the network, which may be an incorrect assumption. 
""")
# TODO: Explain that attention is permutation-invariant by default — 
# it treats the input as a *set*, not a sequence.
# Without positional info, "the cat sat on the mat" and any permutation of 
# those tokens produce the same attention outputs.
# Point to Vaswani et al. 2017 — https://arxiv.org/abs/1706.03762
st.divider()

st.header("Conditions for Positional Encoding")

st.markdown("""
Any good positional encoding scheme must satisfy three conditions. For anyone seeking a much deeper mathematical treatment, kindly refer Wang et. al 2020, [ENCODING WORD ORDER IN COMPLEX EMBEDDINGS](https://openreview.net/pdf?id=Hke-WTVtwr)

- **Preserves absolute positional information** - Each token position must be independently identifiable
- **Preserves relative positional information** - As was noted in the paper, positional values separated by any offset should be represented as a linear function of the original positional value 
- **Length generalization** - The encoding scheme must extend beyond whatever length the model was trained for. For instance, while training million token context windows, we don't necessarily train the model for that length owing to lack of quality data and compute costs. We expect the model to be able to understand with smaller sample lengths. If we chose an encoding scheme that drops off to zero by the 1000th token, the model will not be able to process the difference between future tokens             
""")

st.header("Types of Positional Encoding")

st.markdown("""
**Absolute Positional Encoding** schemes inject information that provide both absolute and relative positional information. The sinusoidal encoding scheme (discussed in next section) from the original transformers paper is an APE. 

**Relative Positional Encoding** only encodes relative token position information. Remember from our [Feedforward Networks Chapter](feedforward#the-formula) that attention focuses on intermixing information from different tokens while FFNs operate on a per token level. Therefore, this relative positional information mostly needs to be tackled in the attention matrix, which is the approach used by papers such as T5 and Transformer XL. RPE is generally considered to be more robust to scaling to unseen lengths. Mind you, each paper makes different architectural choices, and there is too much to cover for the scope of this tutorial. The interested reader is directed to this [survey paper](https://arxiv.org/html/2312.17044v4)
            
There is another separation, i.e. whether the PEs are learned or static. Learned positional encoding schemes use trained params that learns the positional encoding. While effective, it immediately poses an obvious problem - the model cannot generalize to new sequence lengths since it was never trained for them. It was used by papers such as BERT, but is generally avoided now. 
""")
# ─── 2. Sinusoidal encoding (original Transformer) ───────────────────────────

st.info("There are a lot of different PE schemes, but to focus our efforts, we will focus only on three in detail - the original sinusoidal PE, Rotary PE, and ALiBI. What is important to understand is that we want to inject positional info into the embeddings so that the permuation invariant attention matrix can learn to attend by position. Each PE takes different parts of the network's properties to exploit")

st.header("Sinusoidal positional encoding")

st.markdown("""
The sinusoidal variant from [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) gives every position a unique "fingerprint" built from sine and cosine waves at geometrically decreasing frequencies:
""")
st.latex(r"""
PE_{(pos,\, 2i)}   = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)
\qquad
PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
""")
st.markdown("""
where $pos$ is the position in the sequence and $i$ indexes the embedding dimension. $d$ is the embedding dimension length. For each dimension in the embedding, we alternate between sin and cosine functions. The numerator encodes the absolute position, while the $i$ in the denominator encodes dimensional information. The reason we divide by $10000^{2i/d}$ is simply to create very low frequency waves - recall that sin and cosine waves are cyclical. We only want to exploit the initial parts where they remain unique. After a complete cycle, the values will begin to repeat. The value 10000 was determined empirically. 
            
Low-frequency waves vary slowly across positions (coarse structure); high-frequency waves
vary quickly (fine structure). Together they tile the embedding space so no two positions
collide. The positional information is provided to the network simply by adding this positional embedding to the original embedding before passing it to the first transformer block. We will see this in action in the next chapter. 

Below, we have an interactive heatmap. Along the y-axis, you can see the positional index, and x-axis provides information about each dimension. Each row serves as one positional embedding.
""")


# ── Sinusoidal heatmap widget ─────────────────────────────────────────────────

st.markdown("**Sinusoidal encoding — interactive heatmap**")

col1, col2 = st.columns([1, 3])
with col1:
    max_pos   = st.slider("Sequence length",  min_value=128, max_value=8192, value=128,  step=16)
    d_model   = st.slider("Model dimension",  min_value=128, max_value=1024, value=128,  step=16)

def sinusoidal_pe(max_pos, d_model):
    PE = np.zeros((max_pos, d_model))
    pos = np.arange(max_pos)[:, None]
    i   = np.arange(d_model)[None, :]
    angles = pos / (10000 ** (2 * (i // 2) / d_model))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE

PE = sinusoidal_pe(max_pos, d_model)

with col2:
    fig = px.imshow(
        PE,
        labels={"x": "Dimension", "y": "Position", "color": "Value"},
        title=f"Sinusoidal PE  ({max_pos} positions × {d_model} dims)",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Each row is the positional vector injected into the embedding for that token position. "
    "Low-frequency (right) dimensions change slowly; high-frequency (left) change rapidly — "
    "together they form a unique fingerprint per position."
)
st.markdown("""
**Key properties**

| Property | Detail |
|---|---|
| Deterministic | Computed analytically — zero learned parameters |
| Linear relative offsets | $PE_{pos+k}$ can be expressed as a fixed linear function of $PE_{pos}$, so attention can learn to attend by relative distance. Reading through the proof provided in the expander is recommended, it also motivates RoPE |
| Limited extrapolation | Works well up to the training sequence length; degrades beyond it |
""")

with st.expander("Proof for preservation of linearity", expanded = False): 
    st.markdown(r"""
        ## Why sinusoidal PE preserves linear relative offsets

        The claim is that there exists a matrix $M_k$ depending only on the offset $k$ — not on $pos$ — such that:

        $$PE_{pos+k} = M_k \cdot PE_{pos}$$

        ---

        ### Setup

        Group the encoding into 2D blocks, one per frequency $\omega_i = \dfrac{1}{10000^{2i/d_{\text{model}}}}$. For dimension pair $i$:

        $$\mathbf{p}_{pos}^{(i)} = \begin{pmatrix} \sin(\omega_i \cdot pos) \\ \cos(\omega_i \cdot pos) \end{pmatrix}$$

        The full $PE_{pos}$ is these blocks stacked — so it suffices to prove the identity blockwise.

        ---

        ### Core: angle addition identities

        Expand $\mathbf{p}_{pos+k}^{(i)}$ using the sine and cosine addition formulas:

        $$\sin(\omega_i(pos + k)) = \sin(\omega_i \cdot pos)\cos(\omega_i \cdot k) + \cos(\omega_i \cdot pos)\sin(\omega_i \cdot k)$$

        $$\cos(\omega_i(pos + k)) = \cos(\omega_i \cdot pos)\cos(\omega_i \cdot k) - \sin(\omega_i \cdot pos)\sin(\omega_i \cdot k)$$

        ---

        ### Rewrite as a matrix multiplication

        Collecting the $pos$-dependent terms:

        $$\mathbf{p}_{pos+k}^{(i)} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega_i \cdot pos) \\ \cos(\omega_i \cdot pos) \end{pmatrix} = M_k^{(i)} \cdot \mathbf{p}_{pos}^{(i)}$$

        $M_k^{(i)}$ is a **rotation matrix** by angle $\omega_i k$. It depends only on $k$ and the fixed frequency $\omega_i$ — never on $pos$.

        ---

        ### Lifting to the full encoding

        Since every dimension pair transforms independently:

        $$PE_{pos+k} = M_k \cdot PE_{pos}$$

        where $M_k$ is block-diagonal:

        $$M_k = \operatorname{diag}\!\left(M_k^{(0)},\ M_k^{(1)},\ \ldots,\ M_k^{(d/2-1)}\right)$$

        ---

        ### Why this matters for attention

        The dot-product attention score between positions $pos$ and $pos + k$ involves $PE_{pos}^\top PE_{pos+k}$:

        $$PE_{pos}^\top PE_{pos+k} = PE_{pos}^\top M_k\, PE_{pos}$$

        Because each $M_k^{(i)}$ is a rotation matrix it is orthogonal $(M_k^\top M_k = I)$, so it preserves norms.
        The score depends on $k$ in a consistent, translation-invariant way — shift the whole sequence by any
        constant and the relative attention pattern is unchanged.
        """)

st.divider()

# ─── 3. Learned absolute encodings (BERT / GPT) ──────────────────────────────

st.header("Learned absolute positional encodings")

st.markdown("""
Learned positional embeddings are just an additional learned embedding matrix for position, as opposed to using a formula. As we mentioned, they do cause one obvious issue - they are not extrapolable to new sequence lengths since the model has never seen these positions before. This was used by BERT and GPT-2, but is largely not used now. 

Interestingly, [this blog post by Lesswrong](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix) shows how GPT-2's learned positional encoding still shows a similar periodicity in a helical structure, even the cosine similarity between the embeddings have a periodic nature, explained by the previously found helical structure.
""")

st.header("Additive Relative positional encodings")

st.markdown("""
In some cases, we use relative positional encodings as added values in the attention matrix by modifying the QK values. The 'additive' part here is basically the notion that the original token embedding and positional embedding are summed. This was used in T5's relative bias, ALiBi, KERPLE, FIRE etc. These may be learned params or static depending on the specific formulation. We leave this to the reader to dive deeper into. 
""")

# ─── 4. RoPE ─────────────────────────────────────────────────────────────────
st.header("Rotary positional encoding (RoPE)")

st.markdown("""
Rotary Positional encoding is a relative positional encoding scheme applied to the $Q$ and $K$ projections instead of adding it in the beginning. The embedding vectors are rotated in space instead of adding a positional vector to it. 

Before diving into the formula, it helps to ask: *why rotate at all?*

Absolute positional encodings assign a fixed embedding to each position, i.e. token 4 always gets the same vector. But what a model really needs is **relational** information: token 7 comes 3 steps after token 4. That relationship is the same no matter where in the sequence it occurs. Relative encodings capture this reusability; absolute encodings don't, at least not naturally.
            
Another thing to note is that as we mentioned, APEs (such as the original sinusoidal embedding) are added directly to the token embeddings. This causes two issues - first, the magnitude of the vector might change, which modifies several underlying properties. Consider the first 100 values of sinusoidal embedding, let us see how it changes the magnitude (L2 Norm) for every twenty steps. 
""")

# Parameters
embedding_dim = st.slider("Embedding Dimension", 16, 512, 128, step=16)
max_position = st.slider("Max Position", 10, 200, 100, step=10)
seed = st.number_input("Random Seed", value=42)

np.random.seed(seed)

# Random word embedding
word_embedding = np.random.randn(embedding_dim)

# Sinusoidal positional encoding
def get_sinusoidal_encoding(position, d_model):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = np.exp(i * -np.log(10000.0) / d_model)
        pe[i] = np.sin(position * div_term)
        if i + 1 < d_model:
            pe[i + 1] = np.cos(position * div_term)
    return pe

positions = list(range(max_position))
magnitudes_original = []
magnitudes_with_pe = []
angle_changes = []

for pos in positions:
    pe = get_sinusoidal_encoding(pos, embedding_dim)
    combined = word_embedding + pe

    # Norms
    orig_norm = np.linalg.norm(word_embedding)
    new_norm = np.linalg.norm(combined)

    magnitudes_original.append(orig_norm)
    magnitudes_with_pe.append(new_norm)

    # Direction change
    cos_sim = np.dot(word_embedding, combined) / (orig_norm * new_norm + 1e-8)
    angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    angle_changes.append(angle)

# Dataframe
df = pd.DataFrame({
    "Position": positions,
    "Original Norm": magnitudes_original,
    "With PE Norm": magnitudes_with_pe,
    "Angle Change (radians)": angle_changes
})

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["Position"], y=df["Original Norm"], mode='lines', name='Original Norm'))
fig1.add_trace(go.Scatter(x=df["Position"], y=df["With PE Norm"], mode='lines', name='With PE Norm'))
fig1.update_layout(xaxis_title="Position", yaxis_title="L2 Norm", title="Effect of Positional Encoding on Magnitude")
#st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Position"], y=df["Angle Change (radians)"], mode='lines', name='Angle Change'))
fig2.update_layout(xaxis_title="Position", yaxis_title="Angle (radians)", title="Directional Shift Due to PE")
#st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1: 
    st.plotly_chart(fig1, use_container_width=True)
with col2:     
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("""
As we can see above, adding positional embeddings to the original embeddings can cause a lot of change in the underlying natuure of the token. We ideally want to avoid this competition, in particular the change in magnitude of the embedding.            

RoPE's insight is that you can encode relative position *for free* via the dot product. If you rotate each query and key vector by an angle proportional to its absolute position, then **Q·K automatically depends only on the difference in angles** — i.e., the relative offset. No extra parameters, no additive encoding that competes with the content, and no change in magnitude, only direction, which preserves underlying semantic content.

### The rotation in 2D

RoPE operates on pairs of dimensions at a time. For a 2D slice of a token's vector at position $m$, with rotation frequency $\\theta$, the transformation is:
""")
st.latex(r"""
f(\mathbf{x},\, m) =
\begin{pmatrix}
x_1 \cos m\theta - x_2 \sin m\theta \\
x_1 \sin m\theta + x_2 \cos m\theta
\end{pmatrix}
""")

st.markdown("""
This is a standard 2D rotation matrix applied to $(x_1, x_2)$ by angle $m\\theta$. For the full $d$-dimensional vector, this is tiled across all $d/2$ pairs simultaneously using a block-diagonal rotation matrix:
""")

st.latex(r"""
f(\mathbf{x},\, m) =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1  & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 &  \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} &  \cos m\theta_{d/2}
\end{pmatrix}
\mathbf{x}
""")

st.markdown("""
The frequencies $\\theta_i$ follow a geometric schedule introduced in the original RoPE paper:
""")

st.latex(r"""
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \dots, \frac{d}{2} - 1
""")

st.markdown("""
At $i = 0$ this gives $\\theta_0 = 1$, the fastest rotation. At $i = d/2 - 1$ it gives a
value close to zero, the slowest rotation. The base value $10000$ is a hyperparameter
inherited from the sinusoidal encodings of Vaswani et al. (2017); modern models often use
much larger bases (LLaMA 3 uses $500{,}000$) to slow all frequencies down and improve
long-context behaviour.
""")

st.markdown("""
In practice, multiplying by this sparse block-diagonal matrix is never done explicitly.
Instead, the rotation is decomposed into an elementwise multiply and an elementwise add,
which is far cheaper to compute and trivial to implement with precomputed sine and cosine
tables:
""")

st.latex(r"""
f(\mathbf{x},\, m) =
\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \\ x_{d-1} \\ x_d \end{pmatrix}
\otimes
\begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix}
+
\begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \\ -x_d \\ x_{d-1} \end{pmatrix}
\otimes
\begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}
""")

st.markdown("""
The $\\otimes$ denotes elementwise multiplication. The cosine terms scale the original
coordinates, the sine terms scale the negated-and-swapped coordinates, and their sum
reproduces the rotation exactly. The sin and cosine values depend only on position $m$ and
the fixed frequencies $\\theta_i$, so they are precomputed once for all positions up to the
maximum sequence length and reused across every forward pass.
""")
st.markdown("""
### Why the dot product encodes relative position

The key property: when you take the inner product of a rotated **query** at position $m$ with a rotated **key** at position $n$, the result depends only on $m - n$, not on $m$ and $n$ individually. The absolute positions cancel. The attention score between any two tokens is a function of *how far apart they are*.

### Key properties

- **No added parameters** — the rotation is applied post-projection with a fixed matrix.
- **Relative distances emerge automatically** from the dot product structure.
- **Better length extrapolation** than absolute methods, especially with extensions like YaRN, which rescales frequencies to reduce out-of-distribution degradation.
- **Widely adopted**: LLaMA, Mistral, Gemma, Phi, Qwen, and most modern open-weight models use RoPE.

*Paper: Su et al. 2021 — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)*
""")

st.info("The original paper has all the relevant math in detail. I strongly recommend reading it, in some ways it provides a post-hoc rationalization for why sinusoidal embeddings worked - similar to RoPE they added the rotation matrix information into the embeddings but that information was added into the embedding vector which in turn competes with the original embedding values")

with st.expander("Intuitive Proof: the dot product depends only on the relative offset"):
    st.markdown("""
        Take two vectors **q** at position $m$ and **k** at position $n$, each rotated by RoPE in a single 2D subspace. They represent the QK value for a singular pair of embeddings:

        $$f(\\mathbf{q}, m) = \\begin{pmatrix} q_1 \\cos m\\theta - q_2 \\sin m\\theta \\\\ q_1 \\sin m\\theta + q_2 \\cos m\\theta \\end{pmatrix}$$

        $$f(\\mathbf{k}, n) = \\begin{pmatrix} k_1 \\cos n\\theta - k_2 \\sin n\\theta \\\\ k_1 \\sin n\\theta + k_2 \\cos n\\theta \\end{pmatrix}$$

        Computing the dot product $f(\\mathbf{q}, m)^\\top f(\\mathbf{k}, n)$:

        $$= (q_1 \\cos m\\theta - q_2 \\sin m\\theta)(k_1 \\cos n\\theta - k_2 \\sin n\\theta)$$
        $$+ (q_1 \\sin m\\theta + q_2 \\cos m\\theta)(k_1 \\sin n\\theta + k_2 \\cos n\\theta)$$

        Expanding and collecting terms, the $\\cos m\\theta \\cos n\\theta$ and $\\sin m\\theta \\sin n\\theta$ factors combine via the identity $\\cos A \\cos B + \\sin A \\sin B = \\cos(A - B)$:

        $$= (q_1 k_1 + q_2 k_2)\\cos(m\\theta - n\\theta) + (q_1 k_2 - q_2 k_1)\\sin(m\\theta - n\\theta)$$

        Both terms depend on $m$ and $n$ only through the difference $m - n$. The absolute positions $m$ and $n$ do not appear individually anywhere in the result. This is the core property RoPE is built on.

        For the full $d$-dimensional case, the same identity applies independently across each of the $d/2$ subspace pairs, with each pair using a different base frequency $\\theta_i = 10000^{-2i/d}$. The derivation and full matrix form are in Su et al. 2021, Section 3.2.
        """)

# ── RoPE rotation visualizer ──────────────────────────────────────────────────

st.markdown("**RoPE — rotation visualizer**")

st.markdown("""
Consider a given embedding vector. The only thing we are doing here, is rotating it to encode positional information. Note that the magnitude of the vector and origin remain unchanged, we are only modifying its rotation in space. Most importantly, you can see that if the positions are the same (i.e. m = n), both vectors point in the same direction, showing how relative information is preserved
""")

col1, col2 = st.columns([1, 3])
with col1:
    base_theta = st.slider("Base θ (×π/64)", min_value=1, max_value=16, value=4)
    pos_a      = st.slider("Position m", min_value=0, max_value=32, value=4)
    pos_b      = st.slider("Position n", min_value=0, max_value=32, value=10)

theta = base_theta * np.pi / 64

def rotate(x, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * x[0] - s * x[1],
                     s * x[0] + c * x[1]])

vec  = np.array([1.0, 0.0])
va   = rotate(vec, pos_a * theta)
vb   = rotate(vec, pos_b * theta)

fig_rope = go.Figure()
for v, name, color in [(vec, "Original", "#999"),
                        (va,  f"pos={pos_a}", "#E74C3C"),
                        (vb,  f"pos={pos_b}", "#3498DB")]:
    fig_rope.add_trace(go.Scatter(
        x=[0, v[0]], y=[0, v[1]],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=3),
        marker=dict(size=[0, 10])
    ))

fig_rope.update_layout(
    title=f"Rotation by position  (θ={theta:.3f} rad/pos) — dot product = {np.dot(va, vb):.3f}",
    xaxis=dict(range=[-1.4, 1.4], scaleanchor="y"),
    yaxis=dict(range=[-1.4, 1.4]),
    height=420,
    legend=dict(orientation="h")
)

with col2:
    st.plotly_chart(fig_rope, use_container_width=True)

st.caption(
    f"The dot product between pos={pos_a} and pos={pos_b} is determined purely "
    f"by their angular gap ({abs(pos_b - pos_a)} steps × θ). "
    "Try moving both positions together — the dot product stays the same."
)

st.divider()

# ─── 5. Attention vs position distance ───────────────────────────────────────

st.header("How attention weight decays with distance")

st.markdown("""
Constrained by quality data and compute, when we train models for million token contexts, we typically don't use only million token length datasets. As we will see later, for long context windows, we train mostly on shorter data then include some samples of longer context lengths, with the expectation that the model extrapolates (generalizes) to previously unseen lengths. It's also how humans process languages, we've never been exposed to all the possible sentence structures on the planet, we understand how language works and construct sentences on the fly. 
            
The underlying nature of the FFNs and other components should be able to tackle the token level information effectively, but the one place where positional encoding plays a huge part is in the attention matrix.             

The intuition is appealing: tokens that are far apart should, on average, attend to each other
less than nearby tokens. RoPE is often cited as enforcing this property by design. The reality
is more nuanced, and worth understanding carefully before drawing conclusions about effective
context length.

### The theoretical upper bound does decay

The argument begins with the structure of the RoPE dot product. Across the $d/2$ frequency
subspaces (remember that when we apply RoPE, we apply it to pairs of embeddings, i.e. there are $d/2$ subspaces of frequencies), each pair contributes an oscillating term. When you sum them, the upper bound on the attention score — the maximum value the dot product *could* take — falls off with relative distance $|m - n|$ at a rate of $O(1/|m-n|)$ (Su et al., 2021, via Abel summation). This is the result that motivated the long-term decay framing.

The multi-frequency structure explains the shape of that decay. High-frequency dimensions
(small $i$, large $\\theta_i$) rotate quickly with position, so even a modest offset sends
them into near-orthogonality. Low-frequency dimensions rotate slowly and maintain correlation
over much larger offsets. Their superposition produces the characteristic decay curve — rapid
at first, then flattening as the slow dimensions dominate.

### What trained models actually learn

The bound on the upper bound is not the same as the typical behavior. A more recent line of work
challenges whether this theoretical property translates into real attention patterns in trained
models.

[Barbero et al. (2024)](https://arxiv.org/pdf/2406.04267) studied Gemma 7B and found that **attention weights do not reliably decay with distance** once the model is trained. They show that RoPE does not, in general, prevent a token from receiving maximal attention at *any* relative distance, and that the model learns to exploit low-frequency dimensions as semantic channels that are largely position-independent. The decay story, they argue, is not the core reason RoPE works.

Empirically, trained LLMs show a **U-shaped attention distribution**: strong local attention, a
trough over mid-range tokens, and a secondary peak at very distant tokens — what is sometimes
called the "attention sink" at early positions. [Chen et al. (2024, HoPE)](https://arxiv.org/html/2410.21216v1) observe this pattern
across multiple models and decompose it to specific frequency components in RoPE that actively
counteract the theoretical decay during training.

### Two kinds of long-term decay

There are actually two separate decay properties:

1. **Attention score decay** — the raw dot product upper bound shrinks with distance.
2. **Discrimination decay** — the ability to distinguish a relevant key from a random one
   degrades with distance, even if absolute scores remain high.

The second property turns out to be the more consequential one for long-context retrieval. A
model may assign a non-trivial attention score to a distant token while becoming *unable to
tell* whether that token is actually relevant. This is what breaks down before the nominal
context length is exhausted.

### Implications for effective context length

These findings matter for interpreting a model's stated context window. A model trained on
128k tokens does not necessarily reason equally well across all 128k positions. The
discrimination decay means retrieval from early in the context degrades before the hard
positional cutoff is reached, which is why benchmarks like RULER and NIAH often reveal a
sharp performance drop well before the advertised limit.

Extending the RoPE base frequency (as done in LLaMA 3, Gemma 2, and others) slows the rotation
of each subspace, pushing the decay curve outward and making mid-range positions better
calibrated — but it does not eliminate the fundamental constraint. Methods like [YaRN](https://arxiv.org/abs/2309.00071) go further
by rescaling frequencies non-uniformly, preserving fast rotations for local structure while
slowing the low-frequency channels responsible for long-range discrimination.
""")

st.markdown("""
AliBi [Press et al. 2021](https://arxiv.org/abs/2108.12409) is a different approach where we use additive, non-learned static positional encodings that explicitly adds a relative position distance penalty and has been shown to improve inference and training accuracy. It was used in several major models like Falcon and MPT
            
These are elements we will cover in greater detail when we dive into long context extension. For now, it is sufficient to remember that positional encodings are what allow attention to discriminate by distance, therefore the encoding scheme has an outsized influence on when and where attention is applied. 
""")

st.divider()

st.header("Implementation")

st.markdown("We will not be providing implementation here, it will be provided directly in the next chapter so that the differences in encoding schemes are apparent")

col1, col2 = st.columns([3, 5])
col1.page_link("pages/architecture/09_transformer_block.py",         label="Next: Transformer Block →")