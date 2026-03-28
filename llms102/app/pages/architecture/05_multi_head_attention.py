# app/pages/architecture/05_multi_head_attention.py

import streamlit as st
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils.code_loader import show_source, run_it_yourself

st.set_page_config(page_title="Multi-Head Attention | llms102", layout="wide")

# ─── load artifacts ───────────────────────────────────────────────────────────

@st.cache_data
def load_artifacts():
    try:
        weights = np.load("artifacts/attention/weights.npy")
        tokens = np.load("artifacts/attention/tokens.npy", allow_pickle=True).tolist()
        return weights, tokens
    except FileNotFoundError:
        return None, None

weights, tokens = load_artifacts()

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Multi-Head Attention")
st.caption("llms102 — the last lesson you will need")

st.header("The limitation of single head attention")

st.markdown("""
A single attention head captures some information, sure, but as it turns out, two heads are better than one, and for attention, many heads are definitely better than one. 
            
In Attention is all you need, the authors implement several heads of the attention we discussed before in parallel. The computation for each attention head is independent and run in parallel, each QKV attending to different subspaces within the same embedding projection, yielding better performance than one alone. 
            
The outputs from all these attention heads are concatenated as a single long tensor, which is then passed through a linear projection layer called the `Output` layer. In mechanistic interpretability papers, this separation of feature concerns becomes important, which we will revisit in later chapters if necessary. 
""")  
st.divider()

# ─── 2. The idea behind multiple heads ───────────────────────────────────────

st.header("Running attention in parallel")

st.markdown("""
Each head has its own QKV projections and calculates attention for itself. Since the projection matrices are learned params, each head attends to different parts of the input embeddings, offering different levels of granularity. 
            
As the paper notes, 
        
> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
            
The outputs are then concatenated, and passed through a linear layer. Since the operations are independent of each other during each head's attention calculations, they can be parallelized for higher throughput.
            
Please make sure to go through all the dimension sizes at each step in the visualization below
""") 

st.latex(r"""
\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)
""")
st.latex(r"""
\text{MultiHeadAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
""")

st.info("""
Please note that in the original paper implementation, the values of `d_k` are derived from `num_heads` and embedding dims. It is often easier to do so, since this means that the concatenation is simpler and makes the output projection just a linear transformation matrix. However, I do not see why we cannot have free `d_k`, the output projection of which will be managed by the output head. I have provided both visualizations, but if I find a solid reason as to why we cannot use free `d_k`, I will update
""")

# 🔘 Mode toggle
mode = st.radio(
    "Mode",
    ["Tied (d_k = d_model / heads)", "Free (independent d_k)"]
)

# 🎛 Controls
d_model = st.slider("d_model", 64, 1024, 256, step=64)
num_heads = st.slider("Number of heads", 1, 16, 4)

if mode == "Free (independent d_k)":
    d_k = st.slider("d_k (per head)", 8, 256, 64, step=8)
else:
    if d_model % num_heads != 0:
        st.error("d_model must be divisible by number of heads")
        st.stop()
    d_k = d_model // num_heads

# Derived
concat_dim = num_heads * d_k

st.markdown(f"""
### Dimensions
- Per-head dimension: **d_k = {d_k}**
- Concatenated dimension: **{num_heads} × {d_k} = {concat_dim}**
- Output projection: **{concat_dim} → {d_model}**
""")

def build_graph(d_model, d_k, n_heads, concat_dim):
    dot = "digraph MHA {\n"
    dot += "rankdir=LR;\n"
    dot += 'node [shape=box style="rounded,filled" fontname="Helvetica"];\n'

    # Input
    dot += f'X [label="Input\\n(seq × {d_model})", fillcolor="#E3F2FD"];\n'

    for i in range(1, n_heads + 1):
        dot += f"""
        subgraph cluster_{i} {{
            label="Head {i}";
            style="rounded,filled";
            color="#90CAF9";
            fillcolor="#E3F2FD";

            Q{i} [label="Wq{i}\\n({d_model}→{d_k})\\nQ: seq×{d_k}", fillcolor="#FFF3E0"];
            K{i} [label="Wk{i}\\n({d_model}→{d_k})\\nK: seq×{d_k}", fillcolor="#FFF3E0"];
            V{i} [label="Wv{i}\\n({d_model}→{d_k})\\nV: seq×{d_k}", fillcolor="#FFF3E0"];

            A{i} [label="Attention\\n(seq×{d_k})", fillcolor="#E8F5E9"];

            Q{i} -> A{i};
            K{i} -> A{i};
            V{i} -> A{i};
        }}
        """

        dot += f"X -> Q{i}; X -> K{i}; X -> V{i};\n"
        dot += f'A{i} -> O{i} [label="Out\\n(seq×{d_k})"];\n'

    # Concat
    dot += f'CONCAT [label="Concat\\n(seq×{concat_dim})", fillcolor="#F3E5F5"];\n'
    for i in range(1, n_heads + 1):
        dot += f"O{i} -> CONCAT;\n"

    # Output projection
    dot += f'WO [label="Wo\\n({concat_dim}→{d_model})", fillcolor="#E1F5FE"];\n'
    dot += f'OUT [label="Output\\n(seq × {d_model})", fillcolor="#E1F5FE"];\n'

    dot += "CONCAT -> WO -> OUT;\n"
    dot += "}"

    return dot


# Render
st.graphviz_chart(build_graph(d_model, d_k, num_heads, concat_dim))

# ─── 6. Implementation ────────────────────────────────────────────────────────

st.header("Implementation")

st.markdown("")  # one sentence — point to the library file

show_source("llms102/llm_lib/architecture/multi_head_attention.py")

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

run_it_yourself("""
from llm_lib.architecture.multi_head_attention import MultiHeadAttention
                
embedding_dim = 128
proj_dim = 32
seq_len = 100
batch_size = 10
num_heads = 8

attention = MultiHeadAttention(embedding_dim, num_heads)

test_tokens = torch.randn(batch_size, seq_len, embedding_dim)

attention(test_tokens).shape
""")

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/06_feedforward.py", label="Next: Feedforward Layers →")

st.markdown("""
**Further reading**
- Vaswani et al. (2017) — [Attention is all you need](https://arxiv.org/abs/1706.03762)
""")