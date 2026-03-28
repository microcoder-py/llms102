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

st.markdown("")  # your intro — 2-3 sentences

st.divider()

# ─── 1. What does the FFN do ─────────────────────────────────────────────────

st.header("1. What does the FFN do?")

st.markdown("")  # study: Vaswani 2017 section 3.3, Elhage et al. 2022 (A Mathematical Framework)
                 # answer: attention mixes information across tokens
                 #         FFN processes each token independently
                 #         attention is communication, FFN is computation
                 #         FFN is where knowledge is stored empirically

st.divider()

# ─── 2. The formula ──────────────────────────────────────────────────────────

st.header("2. The formula")

st.markdown("")  # one sentence — what goes in, what comes out

st.latex(r"\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2")

st.markdown("")  # answer: what W_1 and W_2 are,
                 #         what the expansion and contraction means,
                 #         why 4x hidden dimension

st.divider()

# ─── 3. Activation functions ─────────────────────────────────────────────────

st.header("3. Activation functions")

st.markdown("")  # study: reason from first principles
                 # answer: why we need nonlinearity at all,
                 #         ReLU — simple, sparse, dying neuron problem
                 #         GELU — smoother, empirically better for transformers
                 #         SwiGLU — used in LLaMA, Mistral, most modern models

# ── activation widget ─────────────────────────────────────────────────────────

st.subheader("Activation functions — interactive")

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

# ─── 4. Dimension expansion ──────────────────────────────────────────────────

st.header("4. Dimension expansion and contraction")

st.markdown("")  # study: reason from first principles
                 # answer: why expand to 4x then contract back,
                 #         what the expansion gives the model — more capacity per token
                 #         parameter count of FFN vs attention

# ── dimension widget ──────────────────────────────────────────────────────────

st.subheader("FFN parameter count — interactive")

d_model = st.slider("d_model", min_value=64, max_value=2048, value=512, step=64)
expansion = st.slider("Expansion factor", min_value=1, max_value=8, value=4)

d_ff = d_model * expansion

w1_params = d_model * d_ff
w2_params = d_ff * d_model
total_params = w1_params + w2_params

col1, col2, col3, col4 = st.columns(4)
col1.metric("d_model", d_model)
col2.metric("d_ff", d_ff)
col3.metric("FFN params", f"{total_params:,}")
col4.metric("% of attention params", f"{total_params / (4 * d_model * d_model) * 100:.0f}%")

st.caption(
    f"W_1 shape: ({d_model}, {d_ff}) = {w1_params:,} params. "
    f"W_2 shape: ({d_ff}, {d_model}) = {w2_params:,} params. "
    f"Total: {total_params:,}. "
    f"At 4x expansion, FFN holds {total_params / (4 * d_model * d_model) * 100:.0f}% "
    f"as many parameters as a single attention layer."
)

st.divider()

# ─── 5. FFN as memory ────────────────────────────────────────────────────────

st.header("5. FFN as memory")

st.markdown("")  # study: Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories
                 #        arxiv 2012.14913
                 # answer: neurons in FFN act as key-value pairs
                 #         W_1 rows are keys — pattern detectors
                 #         W_2 columns are values — what to add to the residual stream
                 #         this is where factual knowledge is stored
                 #         editing model knowledge = editing FFN weights

st.markdown("Geva et al. (2021) — [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)")

st.divider()

# ─── 6. Implementation ───────────────────────────────────────────────────────

st.header("6. Implementation")

st.markdown("")  # one sentence

show_source("llm_lib/architecture/feedforward.py")

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

run_it_yourself("""
from llm_lib.architecture.feedforward import FeedForward
import torch

ffn = FeedForward(d_model=512, expansion=4)
x = torch.randn(1, 10, 512)  # (batch, seq_len, d_model)
output = ffn(x)
print(output.shape)           # (1, 10, 512)
""")

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/07_layer_norm_residuals.py", label="Next: Layer Norm & Residuals →")

st.markdown("""
**Further reading**
- Vaswani et al. (2017) — [Attention is all you need](https://arxiv.org/abs/1706.03762)
- Geva et al. (2021) — [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
""")