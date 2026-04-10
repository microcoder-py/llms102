# app/pages/architecture/04_attention.py

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from utils.code_loader import show_source, run_it_yourself
st.set_page_config(page_title="Attention | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Attention")
st.caption("Neural networks have trouble focusing")

st.markdown("""
Imagine you were reading a book or a research paper, while you're extremely sleepy. It tends to happen that we take the words in, but a page in, and we realise we remember nothing of the last page. 
            
Guess what, even neural networks had that problem! Earlier language models used one or the other form of Recurrent Neural Networks (Karpathy even wrote an article titled [The unreasonable effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)). We won't be going into the reasons as to why RNNs were ultimately replaced by transformer networks, but briefly, 
            
 - **The vanishing/exploding gradient problem**: Training RNNs on long sequences often meant that gradients would either zero out, or explode, destabilizing training. And when I say long sequences, I am not even talking about the present day 1 million token context window, think about at least two orders of magnitude smaller 
- **Fixed window compression**: RNNs use a fixed window to process incoming sequences, and each time they slide over to the next token, they compress the past info in what they call a hidden state. Over multiple steps, the information sharing between two tokens very far apart is very small, making it hard to capture long term dependencies. 
- **Parallelization**: RNNs process sequences step by step, making parallelization incredibly hard
            
Given these constraints, there was a need to overhaul the entire architecture for processing any data that had long term sequential dependencies. The proposed architecture comes from Vaswani et al. (2017) — [Attention is all you need](https://arxiv.org/abs/1706.03762), which we call the transformer network, today the de facto standard for almost all models. 
""")

with st.expander("Further Reading", expanded = False):
    st.markdown("""
        We do not deal with the specific of RNNs, because for the most part they are not totally relevant to our discussion here apart from motivating why we needed an architectural overhaul. The interested readers may look up the relevant terms, and read the following 
        - Sutskever et al. (2014) — [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
        - Bahdanau et al. (2014) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
    """)

st.divider()

# ─── 1. The problem attention solves ─────────────────────────────────────────

st.header("The problem attention solves")

st.markdown("""
Attention provides a method for determining how much different tokens need to look at each other for context. In particular, with transformers, the formulation allows for arbitrary length sequences to attend to different tokens, solving the fixed window compression problem with RNNs. 

As an example, if I gave you this input data: 

`The clock's big hand sat between 2 and 3. Male lions have manes` 

Then asked you to answer `What hour is it` based on the input, you can summarily discard the second sentence because it offers no information about the time, but the first sentence needs to be analyzed. Below, you can see a simple way in which we apply attention as a graph. 
""")
import networkx as nx
import plotly.graph_objects as go

sentence = "What hour is it? The clock's big hand sat between 2 and 3. Male lions have manes."
words = sentence.split()

# edges — (query token, context token, weight)
# only show meaningful attention above threshold
edges = [
    ("What", "clock's", 0.18),
    ("What", "hand", 0.14),
    ("What", "2", 0.18),
    ("What", "3.", 0.18),
    ("What", "between", 0.06),
    ("hour", "clock's", 0.18),
    ("hour", "hand", 0.14),
    ("hour", "2", 0.18),
    ("hour", "3.", 0.18),
    ("is", "clock's", 0.18),
    ("is", "2", 0.18),
    ("is", "3.", 0.18),
    ("it?", "clock's", 0.18),
    ("it?", "hand", 0.14),
    ("it?", "2", 0.18),
    ("it?", "3.", 0.18),
]

threshold = st.slider("Attention threshold", min_value=0.05, max_value=0.25, value=0.10, step=0.01)
filtered_edges = [(s, t, w) for s, t, w in edges if w >= threshold]

G = nx.DiGraph()
query_nodes = ["What", "hour", "is", "it?"]
context_nodes = ["The", "clock's", "big", "hand", "sat", "between", "2", "and", "3.", "Male", "lions", "have", "manes."]

G.add_nodes_from(query_nodes)
G.add_nodes_from(context_nodes)
for s, t, w in filtered_edges:
    G.add_edge(s, t, weight=w)

# layout — query on top, context on bottom
pos = {}
for i, node in enumerate(query_nodes):
    pos[node] = (i * 2, 3)
for i, node in enumerate(context_nodes):
    pos[node] = (i * 1.2, 0)
# edges
edge_traces = []
for s, t, w in filtered_edges:
    x0, y0 = pos[s]
    x1, y1 = pos[t]
    edge_traces.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode="lines",
        line=dict(width=w * 10, color=f"rgba(66, 133, 244, {w * 3})"),
        hoverinfo="none"
    ))

# nodes
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_colors = ["#FF6B6B" if n in query_nodes else "#4ECDC4" for n in G.nodes()]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=list(G.nodes()),
    textposition=["top center" if n in query_nodes else "bottom center" for n in G.nodes()],
    textfont=dict(size=11),
    marker=dict(size=20, color=node_colors),
    hoverinfo="skip"
)
fig = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title="Attention as a directed graph — query tokens attending to context",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Red nodes — query tokens. Teal nodes — context tokens. "
    "Edge thickness represents attention weight. "
    "Adjust the threshold slider to filter weak connections."
)


# ─── Self Attention ────────────────────────────────────────────────────

st.header("Self Attention")

st.markdown("""
There are several variants of attention. For instance, in the example we saw above, we had a separate question, and we then calculated the tokens it needs to attend to in the input sequence to help it answer the question. This is called **cross-attention**, where we attend to tokens between two sequences. 
            
In **self attention**, the tokens apply attention intra sequence. This is also the reason why we do not need to encode a prompt, input and query separately when talking to tools like Claude or ChatGPT - everything is encoded in one single string, which is then passed to the model, and the model figures out which tokens to attend to in order to find the task to execute and relevant data. 
""")  

prompt = "The clock's big hand sat between 2 and 3. Male lions have manes. What hour is it?"

st.markdown(f"""
Here is a dummy example of what would happen if we use our previous example as an input prompt, tokenized by whitespace splits
            
INPUT: `{prompt}`

As you can see, the tokens of the question attend much more heavily to the relevant information sequence, and much less to the information on lion's manes. 
""")

# Tokenization
tokens = prompt.replace(".", " .").replace("?", " ?").split()

n = len(tokens)
attention_weights = np.zeros((n, n))

# Identify segments
first_sentence_end = tokens.index(".")
second_sentence_end = len(tokens) - tokens[::-1].index(".") - 1
question_start = second_sentence_end + 1

# Build attention manually (strong, abrupt focus)
rng = np.random.default_rng(42)
for i in range(n):
    for j in range(n):
        # If token is part of the question
        if i >= question_start:
            # VERY strong attention to first sentence
            if j <= first_sentence_end:
                attention_weights[i, j] = rng.uniform(5.0, 10.0)
            # Near-zero attention to second sentence
            elif j < question_start:
                attention_weights[i, j] = rng.uniform(0.0, 0.05)
            # Minimal self/question attention
            else:
                attention_weights[i, j] = rng.uniform(0.05, 0.1)
        else:
            # Keep other tokens mildly local
            if abs(i - j) <= 2:
                attention_weights[i, j] = rng.uniform(0.5, 1.0)
            else:
                attention_weights[i, j] = rng.uniform(0.0, 0.1)

# Normalize rows (softmax-like)
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

# Plot with Plotly
fig = px.imshow(
    attention_weights,
    x=tokens,
    y=tokens,
    color_continuous_scale="viridis",
    labels=dict(x="Key (attended to)", y="Query (attending)", color="Attention")
)

fig.update_layout(
    xaxis_tickangle=-90,
    height=700
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Bear in mind that the above is a highly exaggerated example with random data suited to explain what attention is doing. This is **NOT** what happens in the actual transformers - attention is dependent on several learned parameters, and post attention there is a lot of processing done, which means that similar to embeddings, the matrices are not directly interpretable. This was only an illustrative example. 
""")

# ─── Intuition  ────────────────────────────────────────────────────

st.header("Atten-hut! There's an intuition on deck.")

st.markdown("""
Let's assume you walk into a library. You know what you are looking for, perhaps information on why bumblebees fly. The only way for you to find out what books contain what information is to use a gigantic chart at the entry which lists all the names of the textbooks, and maybe a gist. 

Your task, is to find information on why bumblebees fly.

Obviously, first, you would skim through all the names of the books and their gists. It's highly unlikely a textbook on bridge construction could provide information on the flight of insects, but a book on flight in general might. A textbook on the study of insects might be the perfect answer, but again, you do not have access to the inner content just yet, so you need to make a best guess about what books to pick out. 

Once you pick out these books, you skim through and find the exact content. By doing so, you have at least found the best guess for what books make sense and which ones don't, avoiding the entire process of reading every book. 
            
Congratulations, you already understand how attention works. 
""")

st.subheader("Query, Key, Value")

st.markdown("""
Above, we had a **Query** - how bumblebees fly. We went through the approximate information available on each book, i.e. the **Key** to make a good guess regarding what to pick out - in a sense, we constructed a likelihood or probability of how much to 'attend' to that title. 
            
Once we had the probability scores of which titles to read based on our query and key pairs, we then used that information to find **Values**, which was basically actually going to the book and extracting what was relevant.

So intuitively, 
            
- Query and Key are used to construct a probability of which items to attend to 
- These probabilities are used to fetch Values from the items

Let's take a quick look at the formula for attention as highlighted in Attention is all you need.  

""")  

st.latex(r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")

st.markdown("""
Don't worry about implementation, we will first look at the intuition here. As you can see, Q and K interact to form a matrix that basically says 'these titles are worth checking out', and then this is softmaxed to generate a probability distribution. `sqrt(dk)` is used for smoothing out distribution, but ignore it for now. 

Armed with this knowledge, we then select the information we require from each item using V 
""")

st.divider()

# ─── 3. Scaled dot product attention ─────────────────────────────────────────

st.header("Scaled dot product attention")

st.markdown("The full formula:")

st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

st.markdown("""
We haven't mentioned what information is passed to calculate this attention. Remember [embeddings from the last chapter](embeddings)? For now, assume that we receive a sequence of token embeddings from the downstream layer, and it is our intention to apply self-attention to the tokens. The embeddings encode the semantic information about the tokens, helping the attention layer understand the gist - kinda like 'reading' the token's information. The whole picture will make more sense when we discuss the entire architecture. 

This part can get a little unintuitive - what do we have for queries and keys here? Well, the honest answer is that it isn't directly interpretable. The formula has been constructed in such a manner that via training, the model learns for itself how to attend where to attend. We never tell the model what the query or the key is, based on the objective and training, it figures out for itself. Hammering the point in again, these matrices are not directly interpretable, they are dependent on learned parameters and are processed further upstream, so the combination of these may or may not result in the ideal attention matrices we demonstrated earlier.
""")  

st.markdown(r"""
First, we take the input embeddings $X$. We have weight matrices $W_q$, $W_k$, and $W_v$ that project the embeddings into different vector spaces, which is where the actual processing happens. Let the sequence length of $X$ be denoted by $N_X$, which is basically the number of tokens in the sequence, and let the embedding dimension size be $d_{emb}$

$$
Q = XW_q
$$

$$
K = XW_k
$$

$$
V = XW_v
$$
            
Typically, the output dimensions of all three are kept the same, that is, 

$$
dim(Q) = dim(K) = dim(V) = d_q = d_k = d_v
$$
            
This is mainly to ensure ease of matrix multiplication, since attention matrices form a large bulk of calculations in transformers, and simpler shapes allow for faster matrix multiplications on GPUs owing to specifics of how it is done on the chip. More on this in later chapters. 
            
Therefore, each projection has dimensions 
            
$$
(N_X, d_{emb}) * (d_{emb}, d_k) = (N_X, d_k)
$$

Now, we calculate the attention scores. This is as simple as matrix multiplying $Q$ and $K^\top$:

$$
\text{scores} = QK^\top
$$
""")

with st.expander("Tricks for looking at matrix multiplications", expanded = False): 
    st.info(r"""
        When we compute $S = QK^\top$, we are performing a matrix multiplication that produces a matrix of attention scores. If $Q \in \mathbb{R}^{n \times d_k}$ and $K \in \mathbb{R}^{n \times d_k}$, then $S \in \mathbb{R}^{n \times n}$.

        Each entry is a dot product:
        $$
        S_{ij} = q_i \cdot k_j
        $$

        So although we write this as a matrix multiplication, it is really computing the similarity between every query and every key. In other words, each row corresponds to one token comparing itself to all other tokens to decide where to attend.

        Developing intuitions like these will help us rewrite steps to optimize matrix computations, as we will see in the [Quantization Aware Training chapter](qat)
    """)

# ── scaling widget ────────────────────────────────────────────────────────────

st.markdown("""
Since all the output dimensions are equal, the calculation is $ (N_X, d_k) * (d_k, N_X) = (N_X, N_X)$ matrix.

We mentioned the ${\sqrt{d_k}}$ term that smooths out the distribution of the attention. Let us now see why this is used. Remember that after this step, we will be applying `softmax` to the whole matrix. 
            
This function in particular is known well for pushing a high level of separation between values, since it creates a probability distribution between 0-1. To avoid excess squishing (and therefore loss) of information, we smoothen out the distribution and make it less 'peaky'. 
            
The value ${\sqrt{d_k}}$ was mentioned to be used for scaling in the paper, but the theory was not explained explicitly. [Some sources](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) suggest it is based on the variance of the attention matrix. Since the matrix contains the dot product between $d_k$ terms, the variance of the matrix scales proportionally to $d_k$. Larger $d_k$ would create excess separation between peaks and lows, so scaling by $\sqrt{d_k}$ is applied. I would advise the reader to find their own sources as well, since I cannot unequivocally verify this at present.

Here is a simple demonstration of what happens when we apply above smoothing to eight tokens. Do try and change the embedding dimensions of the projections.             
""")  

d_k = st.slider("d_k (embedding dimension per head)", min_value=8, max_value=512, value=64, step=8)

np.random.seed(42)
raw_scores = np.random.randn(8)
scaled_scores = raw_scores / np.sqrt(d_k)

softmax = lambda x: np.exp(x) / np.exp(x).sum()
raw_weights = softmax(raw_scores)
scaled_weights = softmax(scaled_scores)

col1, col2 = st.columns(2)

fig1 = px.bar(
    x=[f"token {i}" for i in range(8)],
    y=raw_weights,
    title="Without scaling — peaked distribution",
    labels={"x": "Token", "y": "Attention weight"},
    color=raw_weights,
    color_continuous_scale="Blues"
)
fig1.update_layout(showlegend=False, coloraxis_showscale=False)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    x=[f"token {i}" for i in range(8)],
    y=scaled_weights,
    title=f"With scaling (÷√{d_k}) — smoother distribution",
    labels={"x": "Token", "y": "Attention weight"},
    color=scaled_weights,
    color_continuous_scale="Blues"
)
fig2.update_layout(showlegend=False, coloraxis_showscale=False)
col2.plotly_chart(fig2, use_container_width=True)

st.caption(
    f"Without scaling, large dot products push softmax into saturation — "
    f"gradients vanish and the model stops learning. "
    f"Dividing by √{d_k} keeps the distribution smooth."
)

st.markdown(r"""
Next, to convert this into a probability distribution, we softmax the output. The reader is invited to look up the softmax function to understand exactly what it does. 

So far, here is what we have 
$$            
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$
            
Let's see a small example of the softmax function in action. Note that `softmax` is applied row-wise, we want to find one token's dependence extent on another independently. Therefore, it is the values per row that form a probability distribution summing to 1. 
""")
col1, col2 = st.columns(2)
with col1:
    min_val = st.slider("Min score", -10.0, 0.0, -2.0)
with col2:
    max_val = st.slider("Max score", 0.0, 10.0, 2.0)

# Generate scores
rng = np.random.default_rng(42)
scores = rng.uniform(min_val, max_val, (3, 3))

# Softmax (row-wise)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

softmax_scores = softmax(scores)

# Layout for plots
col3, col4 = st.columns(2)

with col3:
    st.subheader("Raw Attention Scores")
    fig1 = px.imshow(
        scores,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    st.subheader("Softmax Output")
    fig2 = px.imshow(
        softmax_scores,
        text_auto=".2f",
        color_continuous_scale="Viridis"
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(r"""
At this point, we have the final distribution regarding which token will attend to which other token within the sequence, but we still need to apply this to the values projection, i.e. $V$. Again, superbly simple - and it gives us our final formula. We simply matrix multiple the softmaxed attention matrix with $V$. 
            
Intuitively, we are extracting information relevant to the current token from the other tokens from $V$, giving us our final output.

Therefore,
""")

st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

st.markdown(r"""
As you can see, at least with the formulation, we have a great way to attend to all tokens at once, and in parallel since we are no longer constrained by a context window. Attention is what makes transformers so powerful compared to architectures of the past. 

As always, this does not mean attention is perfect. In further chapters such as [KV Cache](kv_cache) and [Flash Attention](flash_attention) we will see how several optimizations are applied only to reduce the cost of attention itself. In fact, Google released a paper on [improving attention inference](https://research.google/blog/sequential-attention-making-ai-models-leaner-and-faster-without-sacrificing-accuracy/) as I was writing out this tutorial, highlighting how even in 2026, nine years after the release of the original paper, we are still trying to make it less expensive. 
""")

st.divider()

st.header("Attention Masking")

st.markdown("""
When executing language modelling, we want to be able to control the flow of information between tokens. For instance, if it is our objective to perform causal language modelling (i.e. each token only depends on itself and past tokens), we want to disallow any attention flow between that token and future tokens. There could also be other cases where we want the model to capture interactions only between specific tokens because of our training methodology. 
            
In such cases, we apply an **Attention Mask**, where we zero out attention between tokens where there should not be any interactions. This is either done by explicitly zeroing out attention scores, or by adding a negative infinite bias to the scores so that during softmax operation they are automatically zeroed out. We will see the specifics of this in each chapter where we apply the masks. 
""")
st.divider()

# ─── 5. Complexity ───────────────────────────────────────────────────────────

st.header("The cost of attention")

st.markdown("""
Let us compute the total number of operations required to compute attention. If you want to be able to understand how we can optimize attention, it is important to go through each calculation carefully. Here is the formula again:
""")

st.latex(r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")

st.markdown(r"""
First we must compute the projections, $Q$, $K$, and $V$. For each, we have a cost of multiplying $(N_X, d_{emb}) * (d_{emb}, d_k)$ values. Multiplication of elements of vectors sized $d_{emb}$ resulting in $d_{emb}$ FLOPs, followed by $d_{emb} - 1$ additions per entry, resulting in approximately another $d_{emb}$ FLOPs, performed for $(N_x, d_k)$ elements total 

$$
2 * N_X * d_{emb} * d_k \text{ FLOPs} 
$$
            
            
Next, we compute the attention scores via $QK^\top$. This produces a matrix of shape $(N_X, N_X)$, where each entry is a dot product between vectors of dimension $d_k$.

Each dot product requires $d_k$ multiplications and $d_k - 1$ additions, which we approximate as $2 d_k$ FLOPs. Since there are $N_X \times N_X$ such entries, the total cost is approximately:

$$
2 * N_X * N_X * d_k = 2 N_X^2 d_k  \text{ FLOPs}
$$

Next, we scale the attention scores by $\frac{1}{\sqrt{d_k}}$, which requires one scalar operation per entry. This results in an additional cost of:

$$
N_X * N_X = N_X^2 \text{ FLOPs}
$$ 

Next, comes the row-wise softmax on the matrix. Since we apply the softmax row-wise, we should first calculate the cost of softmax for one row sized $N_X$, then add it for $N_X$ rows. 
            
The formula for softmax is 

$$
 \mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}           
$$
            
Therefore, for each row, we have $N_X$ exponentiation operations, $N_X - 1$ additions, and $N_X$ divisions, resulting in a total of approximately 
            
$$
    3 * N_X \text{ FLOPs}
$$

Summing over all rows, we get 

$$
3 * N_X * N_X = 3 N_X^2 \text{ FLOPs}
$$
            
Finally, we have the matrix multiplication with $V$ of dimensions $(N_X, d_k)$. Out attention matrix is of dimensions $(N_X, N_X)$, so for each output element, we need to perform $N_X$ multiplications, and $N_X - 1$ additions, repeated for $(N_X, d_k)$ vectors, resulting in total 

$$
2 * N_X * N_X * d_k = 2N_X^2d_k \text{ FLOPs}
$$
""")  

st.markdown(r"""
To summarize, here is the table and the total cost per step

| Step | Operation | FLOPs |
|------|----------|-------|
| Projections ($Q, K, V$) | $XW_q,\; XW_k,\; XW_v$ | $6 N_X d_{emb} d_k$ |
| Attention Scores | $QK^\top$ | $2 N_X^2 d_k$ |
| Scaling | $\frac{1}{\sqrt{d_k}}$ | $N_X^2$ |
| Softmax | row-wise | $3 N_X^2$ |
| Weighted Sum | $AV$ | $2 N_X^2 d_k$ |
            
Resulting in a sum total of 

            
$$
6 N_X d_{emb} d_k + 4 N_X^2 d_k + 4 N_X^2 \text{ FLOPs}
$$
            
Typically, we keep $d_k$ much smaller than $d_{emb}$, so from the above equation, it becomes clear that attention scales quadratically with $N_X$, meaning that the more the prompt length, the amount of compute required goes up quadratically. 
            
There is one thing I have not yet highlighted - per transformer block, we have multiple attention heads (we will discuss this in depth in further lessons), and multiple such transformer blocks (or layers). 

To see how this becomes especially problematic - an [article by Shashank Shekhar](https://www.shashankshekhar.com/blog/flashmla/flashmla-1-mla#why-attention-scales-on2-with-sequence-length) quotes the following:
""")

st.info("""
The quadratic scaling is brutal. At DeepSeek-V2 scale with 128K context, a single attention layer requires over 1 PFLOP of compute and would take over 5 seconds on an A100 — and there are 60 such layers. Even at 4K context, each layer burns 1.1 TFLOP. This is clearly untenable.
""")

# ── complexity widget ─────────────────────────────────────────────────────────

st.subheader("Attention FLOPs — Interactive")

st.markdown("""
We are showing the number of FLOPs needed for a single attention layer. As we mentioned, there are several such attention heads per transformer layer, and several such transformer layers. However, the below visualisation should help you see how much sequence length affects attention computations. 

For reference, H100, one of the most advanced GPUs on the planet for AI inference at present provides approximately 990 Tera-FLOPs at 32 bit precision. Across all the layers and attention, the costs will quickly add up, and we haven't even started talking about the other layers that will also require compute
""")

col1, col2 = st.columns(2)
with col1:
    max_seq = st.slider("Max sequence length", 128, 128000, 8192, step=128)
with col2:
    d_emb = st.slider("Embedding dimension (d_emb)", 128, 4096, 512, step=128)

# Sequence range
seq_lengths = np.arange(128, max_seq + 1, 128)

# d_k values (powers of 2 up to 16384)
dk_values = [2**i for i in range(5, 14)]  # 32 → 16384

# Compute FLOPs (full formula)
data = []
for dk in dk_values:
    for n in seq_lengths:
        flops = (
            6 * n * d_emb * dk +
            4 * (n ** 2) * dk +
            4 * (n ** 2)
        )
        data.append({
            "Sequence Length": n,
            "FLOPs": flops,
            "d_k": f"{dk}"
        })

# Plot
fig = px.line(
    data,
    x="Sequence Length",
    y="FLOPs",
    color="d_k",
    title="Attention FLOPs Scaling (Full Cost)"
)

fig.update_layout(height=550)

st.plotly_chart(fig, use_container_width=True)
st.divider()
st.header("Code Implementation")
st.markdown("The code is fairly self explanatory, just make sure you understand the shapes correctly. It simply implements the formula")
show_source("llms102/llm_lib/architecture/attention.py")

run_it_yourself("""
from llm_lib.architecture.attention import SelfAttention

embedding_dim = 128
proj_dim = 32
seq_len = 100
batch_size = 10

attention = Attention(embedding_dim, proj_dim)

test_tokens = torch.randn(batch_size, seq_len, embedding_dim)

attention(test_tokens).shape
""")

st.header("Conclusion")

st.markdown("""
Attention is what makes transformers like ChatGPT, Claude, Gemini etc. not just possible, but capable. Their capacity to simultaneously look at tokens that are extremely far apart, even hundreds of thousands of tokens, is what allows them to reason at all, not just reason over long sequences. 
            
However, they are a double edged sword. Cost scales quickly, and we don't have enough GPUs in the world to cater to it. There are further optimizations that are performed to bring these costs down which we will discuss later. 
            
For now, make absolutely sure you understand the fundamentals here, because without a solid grasp of attention, the rest of the network won't make much sense. 
            
Which is to say, pay attention to attention. I will see myself out, thanks. 
""")

# ─── Footer ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/05_multi_head_attention.py", label="Next: Multi-Head Attention →")
