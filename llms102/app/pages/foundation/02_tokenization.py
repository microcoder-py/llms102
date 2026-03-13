# app/pages/foundation/02_tokenization.py

import streamlit as st
import numpy as np
import plotly.express as px
from transformers import GPT2Tokenizer

st.set_page_config(page_title="Tokenization | llms102", layout="wide")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
]

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Tokenization")
st.caption("llms102 — the last lesson you will need")

st.markdown("")  # 2 sentences — what this page covers

st.divider()

# ─── 1. What is tokenization? ────────────────────────────────────────────────

st.header("1. What is tokenization?")

st.markdown("")  # study: any tokenizer explainer
                 # answer: what problem tokenization solves, what a token is

st.divider()

# ─── 2. Character, word, subword ─────────────────────────────────────────────

st.header("2. Character, word, and subword tokenization")

st.markdown("")  # study: HuggingFace tokenizer docs
                 # answer: tradeoffs of each, why subword won

col1, col2, col3 = st.columns(3)

col1.markdown("**Character-level**")
col1.markdown("")  # one line — the failure mode

col2.markdown("**Word-level**")
col2.markdown("")  # one line — the failure mode

col3.markdown("**Subword**")
col3.markdown("")  # one line — why it works

st.divider()

# ─── 3. Byte Pair Encoding ───────────────────────────────────────────────────

st.header("3. Byte Pair Encoding — Sennrich et al. (2016)")

st.markdown("")  # study: Sennrich et al. 2016 arxiv 1508.07909
                 # answer: how BPE builds vocabulary iteratively, what a merge rule is

# ── BPE merge widget ──────────────────────────────────────────────────────────

st.subheader("BPE merges — interactive")

st.markdown("")  # one sentence directing the reader

word = st.text_input("Enter a word", value="tokenization")
chars = list(word)

steps = []
current = list(word)
steps.append(("Initial split", list(current)))

pairs = {}
for i in range(len(current) - 1):
    pair = (current[i], current[i+1])
    pairs[pair] = pairs.get(pair, 0) + 1

if pairs:
    best = max(pairs, key=pairs.get)
    merged = []
    i = 0
    while i < len(current):
        if i < len(current) - 1 and (current[i], current[i+1]) == best:
            merged.append(current[i] + current[i+1])
            i += 2
        else:
            merged.append(current[i])
            i += 1
    steps.append((f"Merge: '{best[0]}' + '{best[1]}'", merged))

for step_name, tokens in steps:
    st.markdown(f"**{step_name}**")
    cols = st.columns(len(tokens))
    for i, (col, token) in enumerate(zip(cols, tokens)):
        col.markdown(
            f"<div style='background:{COLORS[i % len(COLORS)]};padding:8px;border-radius:4px;text-align:center;color:black'>{token}</div>",
            unsafe_allow_html=True
        )
    st.markdown("")

st.markdown("""
> Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of 
> Rare Words with Subword Units.* ACL 2016.
> [arxiv 1508.07909](https://arxiv.org/abs/1508.07909)
""")

st.divider()

# ─── 4. WordPiece & SentencePiece ────────────────────────────────────────────

st.header("4. WordPiece & SentencePiece")

st.markdown("")  # study: Kudo & Richardson 2018 arxiv 1808.06226
                 # answer: how WordPiece differs from BPE, what SentencePiece adds, which models use which

st.markdown("""
> Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and language 
> independent subword tokenizer.*
> [arxiv 1808.06226](https://arxiv.org/abs/1808.06226)
""")

st.divider()

# ─── 5. Vocabulary size ──────────────────────────────────────────────────────

st.header("5. Vocabulary size and its consequences")

st.markdown("")  # study: reason from first principles
                 # answer: too small vs too large, effect on embedding table size

# ── vocabulary size widget ────────────────────────────────────────────────────

st.subheader("Vocabulary size vs embedding table — interactive")

col1, col2 = st.columns(2)
vocab_size = col1.slider("Vocabulary size", min_value=1000, max_value=100000, value=50000, step=1000)
embedding_dim = col2.slider("Embedding dimension", min_value=64, max_value=4096, value=768, step=64)

params = vocab_size * embedding_dim
size_mb = (params * 4) / (1024 ** 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total parameters", f"{params:,}")
col2.metric("Size (float32)", f"{size_mb:.1f} MB")
col3.metric("% of 125M model", f"{(params / 125e6) * 100:.1f}%")

st.divider()

# ─── 6. Tokenization artifacts ───────────────────────────────────────────────

st.header("6. Tokenization artifacts")

st.markdown("")  # study: observe a real tokenizer
                 # answer: whitespace sensitivity, numbers, code, multilingual

# ── token visualizer widget ───────────────────────────────────────────────────

st.subheader("Token visualizer — interactive")

st.markdown("")  # one sentence directing the reader

user_text = st.text_area(
    "Enter any text",
    value="The quick brown fox jumped over the lazy dog. 1 + 1 = 2.",
    height=100
)

if user_text:
    tokens = tokenizer.encode(user_text)
    token_strings = [tokenizer.decode([t]) for t in tokens]

    st.markdown("**Tokens:**")
    html = ""
    for i, token in enumerate(token_strings):
        color = COLORS[i % len(COLORS)]
        html += f"<span style='background:{color};padding:2px 6px;border-radius:3px;margin:2px;display:inline-block;color:black'>{repr(token)}</span>"
    st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(user_text))
    col2.metric("Tokens", len(tokens))
    col3.metric("Chars per token", f"{len(user_text)/len(tokens):.2f}")

st.divider()

# ─── 7. What the model never sees ────────────────────────────────────────────

st.header("7. What the model never sees")

st.markdown("")  # study: reason from first principles
                 # answer: model sees ids not text, what is lost, why tokenization affects capabilities

example = "Hello, world!"
token_ids = tokenizer.encode(example)

st.markdown(f"**Input:** `{example}`")
st.markdown(f"**Token IDs:** `{token_ids}`")
st.markdown(f"**Vocabulary size:** `{tokenizer.vocab_size:,}`")

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 5])
col1.page_link("pages/foundation/03_embeddings.py", label="Next: Embeddings →")

st.markdown("""
**Further reading**
- Sennrich et al. (2016) — [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Kudo & Richardson (2018) — [SentencePiece](https://arxiv.org/abs/1808.06226)
- Kudo (2018) — [Subword Regularization](https://arxiv.org/abs/1804.10959)
- [HuggingFace Tokenizer Docs](https://huggingface.co/docs/transformers/tokenizer_summary)
""")