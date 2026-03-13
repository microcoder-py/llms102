# app/pages/foundation/03_embeddings.py

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Embeddings | llms102", layout="wide")

# ─── synthetic embeddings ─────────────────────────────────────────────────────

np.random.seed(42)
VOCAB_SIZE = 50257
EMBED_DIM = 768
embedding_table = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32)

WORD_IDS = {
    "king": 3364, "queen": 16871, "man": 582, "woman": 2415,
    "paris": 6342, "london": 3576, "france": 4881, "england": 5459,
    "cat": 3797, "dog": 3290, "fish": 3869, "bird": 5827,
    "hello": 31373
}

def get_embedding(word):
    token_id = WORD_IDS.get(word.lower())
    if token_id is None:
        token_id = hash(word) % VOCAB_SIZE
    return embedding_table[token_id]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Embeddings")
st.caption("llms102 — the last lesson you will need")

st.markdown("")  # 2 sentences — what this page covers

st.divider()

# ─── 1. What are embeddings? ─────────────────────────────────────────────────

st.header("1. What are embeddings?")

st.markdown("")  # study: Bengio 2003 section 2, Mikolov 2013 intro
                 # answer: why discrete tokens need continuous representations,
                 #         what an embedding vector is,
                 #         what it means for two tokens to be close

st.divider()

# ─── 2. The embedding table ──────────────────────────────────────────────────

st.header("2. The embedding table")

st.markdown("")  # study: reason from first principles
                 # answer: what the embedding table is,
                 #         how a token id maps to a vector,
                 #         why it is learned not designed

# ── embedding table lookup widget ────────────────────────────────────────────

st.subheader("Embedding table lookup — interactive")

st.markdown("")  # one sentence directing the reader

token_input = st.text_input("Enter a word", value="hello")

if token_input:
    token_id = WORD_IDS.get(token_input.lower(), hash(token_input) % VOCAB_SIZE)
    embedding = embedding_table[token_id]

    col1, col2 = st.columns([1, 3])
    col1.metric("Token ID", token_id)
    col1.metric("Embedding dim", EMBED_DIM)

    fig = px.bar(
        x=list(range(64)),
        y=embedding[:64],
        labels={"x": "Dimension", "y": "Value"},
        title=f"First 64 dimensions of embedding for '{token_input}'",
        color=embedding[:64],
        color_continuous_scale="RdBu"
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    col2.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── 3. Word2Vec ─────────────────────────────────────────────────────────────

st.header("3. Word2Vec — Mikolov et al. (2013)")

st.markdown("")  # study: arxiv 1301.3781
                 # answer: what Word2Vec learned that changed everything,
                 #         skip-gram vs CBOW at a high level,
                 #         why similar words end up close in embedding space

# ── 2D embedding projection widget ───────────────────────────────────────────

st.subheader("Embedding space — interactive")

st.markdown("")  # one sentence directing the reader

sample_words = [
    "king", "queen", "man", "woman",
    "paris", "london", "france", "england",
    "cat", "dog", "fish", "bird"
]

vecs = np.array([embedding_table[WORD_IDS[w]] for w in sample_words])
pca = PCA(n_components=2)
reduced = pca.fit_transform(vecs)

fig = px.scatter(
    x=reduced[:, 0],
    y=reduced[:, 1],
    text=sample_words,
    title="Token embeddings projected to 2D (PCA)",
    labels={"x": "PC1", "y": "PC2"}
)
fig.update_traces(textposition="top center", marker=dict(size=10))
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation 
> of Word Representations in Vector Space.*
> [arxiv 1301.3781](https://arxiv.org/abs/1301.3781)
""")

st.divider()

# ─── 4. From word embeddings to token embeddings ─────────────────────────────

st.header("4. From word embeddings to token embeddings")

st.markdown("")  # study: GPT-2 paper, reason from first principles
                 # answer: how modern LLMs embed subword tokens not words,
                 #         why embedding layer is first in the transformer,
                 #         how embedding dimension relates to model size

col1, col2, col3, col4 = st.columns(4)
col1.metric("GPT-2 Small", "768 dim")
col2.metric("GPT-2 Medium", "1024 dim")
col3.metric("GPT-2 Large", "1280 dim")
col4.metric("GPT-2 XL", "1600 dim")

st.divider()

# ─── 5. Embedding geometry ───────────────────────────────────────────────────

st.header("5. Embedding geometry")

st.markdown("")  # study: Mikolov 2013 analogy paper arxiv 1309.4168
                 # answer: what vector arithmetic means,
                 #         why king - man + woman ≈ queen,
                 #         what this tells us about what embeddings capture

# ── vector arithmetic widget ──────────────────────────────────────────────────

st.subheader("Vector arithmetic — interactive")

st.markdown("")  # one sentence directing the reader

col1, col2, col3 = st.columns(3)
word_a = col1.text_input("Word A", value="king")
word_b = col2.text_input("Word B (subtract)", value="man")
word_c = col3.text_input("Word C (add)", value="woman")

if word_a and word_b and word_c:
    vec_a = get_embedding(word_a)
    vec_b = get_embedding(word_b)
    vec_c = get_embedding(word_c)
    result = vec_a - vec_b + vec_c

    similarities = []
    for word, token_id in WORD_IDS.items():
        sim = cosine_similarity(result, embedding_table[token_id])
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    st.markdown(f"**{word_a} - {word_b} + {word_c} ≈**")
    for word, score in similarities[:5]:
        st.markdown(f"`{word}` — cosine similarity: `{score:.3f}`")

st.markdown("""
> Mikolov, T., Yih, W., & Zweig, G. (2013). *Linguistic Regularities in 
> Continuous Space Word Representations.*
> [arxiv 1309.4168](https://arxiv.org/abs/1309.4168)
""")

st.divider()

# ─── 6. Context changes everything ───────────────────────────────────────────

st.header("6. Context changes everything")

st.markdown("")  # study: BERT intro arxiv 1810.04805
                 # answer: why static embeddings are limited,
                 #         what contextual embeddings are,
                 #         how the same token gets different vectors in different contexts

# ── contextual embedding widget ───────────────────────────────────────────────

st.subheader("Context changes embeddings — interactive")

st.markdown("""
Static embeddings assign a single fixed vector to each token regardless of 
context. The widget below simulates how context shifts a token's representation 
by adding a context vector to the base embedding.
""")

sentence_1 = st.text_input("Sentence 1", value="The bank by the river was muddy.")
sentence_2 = st.text_input("Sentence 2", value="The bank approved my loan.")
target_word = st.text_input("Target word", value="bank")

if sentence_1 and sentence_2 and target_word:
    base_emb = get_embedding(target_word)

    np.random.seed(len(sentence_1))
    context_1 = np.random.randn(EMBED_DIM).astype(np.float32) * 0.3
    np.random.seed(len(sentence_2))
    context_2 = np.random.randn(EMBED_DIM).astype(np.float32) * 0.3

    contextual_1 = base_emb + context_1
    contextual_2 = base_emb + context_2

    static_sim = cosine_similarity(base_emb, base_emb)
    contextual_sim = cosine_similarity(contextual_1, contextual_2)

    col1, col2 = st.columns(2)
    col1.metric("Static similarity", f"{static_sim:.3f}", help="Same vector regardless of context")
    col2.metric("Contextual similarity", f"{contextual_sim:.3f}", help="Different vector per context")

    st.caption(
        "In a real transformer, attention layers shift each token's representation "
        "based on every other token in the sequence. The same token 'bank' gets "
        "a different vector in a financial context versus a geographical one."
    )

st.divider()

# ─── 7. What embeddings are not ──────────────────────────────────────────────

st.header("7. What embeddings are not")

st.markdown("")  # study: reason from first principles
                 # answer: embeddings are not meaning,
                 #         they are not fixed,
                 #         they are learned compression of co-occurrence statistics

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 5])
col1.page_link("pages/architecture/04_attention.py", label="Next: Attention →")

st.markdown("""
**Further reading**
- Mikolov et al. (2013) — [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013) — [Linguistic Regularities in Word Representations](https://arxiv.org/abs/1309.4168)
- Devlin et al. (2018) — [BERT](https://arxiv.org/abs/1810.04805)
- Bengio et al. (2003) — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
""")