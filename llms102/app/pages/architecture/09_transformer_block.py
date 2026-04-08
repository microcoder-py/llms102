import streamlit as st
from utils.code_loader import show_source, run_it_yourself
st.set_page_config(page_title="The Transformer Architecture | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("The Transformer Architecture")
st.caption("Where everything comes together")

st.markdown("""
Now that we have covered all the independent blocks in the transformer network, it is time for us to put it all together. Follow everything step by step, with the background theory in mind. I recognize there are several variations we can use to build the network since each component has its own variations, so I will try my best to highlight them as cleanly as I can. 
            
We did write out the implementations for some of the subcomponents, but it is always recommended to use standard implementations from mature libraries such as PyTorch. They include several different nuances we may have glossed over for the sake of simplicity. 
            
Please note that we will not be performing any training here. What we first need to do is put together the transformer network. The specifics of training are dependent on what we intend to achieve with the model, bbut the underlying architecture remains largely the same. 
""")

st.divider()

st.header("Tokenization and Embedding")

st.markdown("""
The incoming content needs to be tokenized and embedded. Since the tokenizer is a separate subsystem, it is not a part of the training here. It is only used to represent the text in a discrete numerical space, which is then projected into continous space using embeddings. Then we add an embedding matrix, which is essentially a lookup table, mapping a tokenizer index to a embedding
""")

def build_transformer_graph(
    mode="full",
    norm="prenorm",
    include_pos=False,
    pos_type="Sinusoidal"
):
    dot = "digraph Transformer {\n"
    dot += "rankdir=TB;\n"
    dot += 'graph [pad="0.5", nodesep="0.8", ranksep="0.3"];\n'
    dot += 'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=18 width=3];\n'
    dot += 'edge [fontname="Helvetica", fontsize=14];\n'

    colors = {
        "input": "#E3F2FD",
        "compute": "#FFF8E1",
        "norm": "#E8F5E9",
        "residual": "#FFEBEE",
        "output": "#F3E5F5",
        "prob": "#E1F5FE",
        "pos": "#FFFDE7",
        "rope": "#E1BEE7"
    }

    def node(name, label, color):
        return f'{name} [label="{label}", fillcolor="{color}"];\n'

    # -------------------------
    # Embedding
    # -------------------------
    if mode in ["embed", "block", "full"]:
        dot += node("TOK", "Tokenizer", colors["input"])
        dot += node("EMB", "Embedding", colors["compute"])
        dot += 'TOK -> EMB [label="text -> tokens"];\n'

        if include_pos and pos_type == "Sinusoidal":
            dot += node("POS", "Sinusoidal Positional Encoding", colors["pos"])
            dot += node("ADD_POS", "+", colors["residual"])

            dot += '{ rank=same; POS ADD_POS; }\n'
            dot += 'EMB -> ADD_POS [label="x"];\n'
            dot += 'POS -> ADD_POS [label="p", color="red", penwidth=2];\n'

            prev = "ADD_POS"
        else:
            prev = "EMB"
    else:
        dot += node("X", "Input", colors["input"])
        prev = "X"

    # -------------------------
    # Transformer Block
    # -------------------------
    if mode in ["block", "full"]:

        # ---------- PRE-NORM ----------
        if norm == "prenorm":

            dot += node("LN1", "LayerNorm", colors["norm"])
            dot += node("ADD1", "+", colors["residual"])
            dot += node("LN2", "LayerNorm", colors["norm"])
            dot += node("FFN", "Feedforward", colors["compute"])
            dot += node("ADD2", "+", colors["residual"])

            if include_pos and pos_type == "RoPE":
                dot += node("QKV", "Linear (Q,K,V)", colors["compute"])
                dot += node("ROPE", "RoPE (Q,K)", colors["rope"])
                dot += node("ATTN", "Attention", colors["compute"])

                dot += f'{prev} -> LN1 [label="x"];\n'
                dot += 'LN1 -> QKV [label="x_norm"];\n'
                dot += 'QKV -> ROPE [label="Q,K"];\n'
                dot += 'ROPE -> ATTN [label="rotated Q,K"];\n'
            else:
                dot += node("ATTN", "Attention", colors["compute"])

                dot += f'{prev} -> LN1 [label="x"];\n'
                dot += 'LN1 -> ATTN [label="x_norm"];\n'

            dot += 'ATTN -> ADD1 [label="attn(x)"];\n'
            dot += 'ADD1 -> LN2 [label="x1"];\n'
            dot += 'LN2 -> FFN [label="x1_norm"];\n'
            dot += 'FFN -> ADD2 [label="ffn(x1)"];\n'

            dot += f'{prev} -> ADD1 [color="red", penwidth=2];\n'
            dot += 'ADD1 -> ADD2 [color="red", penwidth=2];\n'

            prev = "ADD2"

        # ---------- POST-NORM ----------
        else:

            dot += node("ADD1", "+", colors["residual"])
            dot += node("LN1", "LayerNorm", colors["norm"])
            dot += node("FFN", "Feedforward", colors["compute"])
            dot += node("ADD2", "+", colors["residual"])
            dot += node("LN2", "LayerNorm", colors["norm"])

            if include_pos and pos_type == "RoPE":
                dot += node("QKV", "Linear (Q,K,V)", colors["compute"])
                dot += node("ROPE", "RoPE (Q,K)", colors["rope"])
                dot += node("ATTN", "Attention", colors["compute"])

                dot += f'{prev} -> QKV [label="x"];\n'
                dot += 'QKV -> ROPE;\n'
                dot += 'ROPE -> ATTN;\n'
            else:
                dot += node("ATTN", "Attention", colors["compute"])
                dot += f'{prev} -> ATTN [label="x"];\n'

            dot += 'ATTN -> ADD1;\n'
            dot += 'ADD1 -> LN1;\n'
            dot += 'LN1 -> FFN;\n'
            dot += 'FFN -> ADD2;\n'
            dot += 'ADD2 -> LN2;\n'

            dot += f'{prev} -> ADD1 [color="red", penwidth=2];\n'
            dot += 'LN1 -> ADD2 [color="red", penwidth=2];\n'

            prev = "LN2"

        # 🔲 Cluster
        cluster_nodes = ["ATTN", "ADD1", "LN1", "FFN", "ADD2", "LN2"]

        if norm == "prenorm":
            cluster_nodes = ["LN1", "ATTN", "ADD1", "LN2", "FFN", "ADD2"]

        if include_pos and pos_type == "RoPE":
            cluster_nodes.insert(1, "QKV")
            cluster_nodes.insert(2, "ROPE")

        dot += f'''
        subgraph cluster_block {{
            label="Transformer Block ({norm})";
            style="rounded";
            color="#CCCCCC";

            {' '.join(cluster_nodes)};
        }}
        '''

    # -------------------------
    # Output Head
    # -------------------------
    if mode in ["head", "full"]:
        dot += node("LOGIT", "Linear (Logits)", colors["output"])
        dot += node("SMAX", "Softmax", colors["prob"])
        dot += node("OUT", "Output Token", colors["output"])

        dot += f'{prev} -> LOGIT;\n'
        dot += 'LOGIT -> SMAX;\n'
        dot += 'SMAX -> OUT;\n'

    dot += "}\n"
    return dot

dot_string = build_transformer_graph(mode="embed")

st.graphviz_chart(dot_string)

st.divider()

st.header("Transformer Block")

st.markdown("""
Now, we add a single transformer block. It contains a Multihead Attention operation, followed by FFN. We've already discussed normalization and residual adding operations, which can either be pre-norm, post norm, or some other variant. For now, let's look at the pre and post norm variants only, and no other. 
            
Within a typical transformer model, you would have several of these transformer blocks, each one's final output serving as the input to the next block
""")

norm = st.segmented_control(
    "Normalization Type",
    ["prenorm", "postnorm"],
    key="norm_segment"
)

dot_transformer_block = build_transformer_graph(mode="block", norm = norm)

st.graphviz_chart(dot_transformer_block)

st.divider()

st.header("LM Head")

st.markdown(r"""
Once all the processing by the transformer blocks is done, we need to project the outputs back into vocabulary space. For this, we use something called a language modelling head, which is essentially a matrix that takes in the output embeddings from the last layer and projects it into a tensor of dimension $(\text{Sequence Length}, \text{Vocabulary Size})$. That is, for each token in the sequence, we generate probabilities for how likely each vocabulary token is, and depending on our generation method, we select the next token. 
            
How are these probabilities generated? Simple. During inference, we apply softmax to the outputs of the LM Head along the Vocabulary dimension, which gives us probabilities of each token. However, please bear in mind, during training, we do not apply this softmax explicitly, we operate on the logits space. The loss function we use, Cross Entropy Loss, internally applies softmax while calculating losses. Follow along with this hardcoded assumption for now, we will see the exact reasoning when we train the model. 
""")

dot_transformer_full = build_transformer_graph(mode="full", norm = norm)

st.graphviz_chart(dot_transformer_full)

st.divider()

st.header("Positional Encoding")

st.markdown(r"""
We have not yet added positional information yet. PE can either be Absolute PE, added to the first embedding layer, or relative PE, which typically operates in the QK space within each attention head. Let's consider sinusoidal and RoPE, of which sinusoidal is added once to the first embedding output, while RoPE operates on each QK matrix of each attention head in MHA 
""")
import streamlit as st

pos_type = st.radio(
    "Positional Encoding Type",
    ["Sinusoidal", "RoPE"],
    horizontal=True,
    key = "pos"
)

dot_transformer_full = build_transformer_graph(
    mode="full", 
    norm = norm, 
    include_pos=True, 
    pos_type= pos_type
)

st.graphviz_chart(dot_transformer_full)

st.divider()

st.header("Implementation")

st.markdown(r"""
Let's start writing out our transformer network! In our case, we will build out the following architecture.  

1. Pre-Norm with RMSNorm Layer
2. RoPE Positional Encoding (Which needs a rewrite of the MHA layer) 

Just follow along with the code below, and read the comments as you come across them 
""")

st.info("Please note that this implementation is incomplete - we have not yet considered attention masking. We will add that when we train individual networks after we have discussed the different variants of language modelling")

show_source("llms102/llm_lib/architecture/transformer.py")

st.divider()

col1, col2 = st.columns([3, 5])
col1.page_link("pages/training/10_loss_perplexity.py", label="Next: Loss and Perplexity →")
