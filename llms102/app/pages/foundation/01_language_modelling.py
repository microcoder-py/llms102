# app/pages/foundation/01_language_modelling.py

import streamlit as st
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Language Modelling | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Language Modelling")
st.caption("llms102 — the last lesson you will need")

st.markdown("""
*Research: Write 2-3 sentences. Answer — what is this page about, 
what will the reader understand by the end, why does it matter.*
""")

st.divider()

# ─── 1. What is a language model? ────────────────────────────────────────────

st.header("1. What is a language model?")

st.markdown("""
*Research: Bengio 2003 introduction. Write 2-3 sentences in plain english. 
Answer — what does a language model actually do, not what it seems to do.*
""")

st.markdown("Formally, given a sequence of tokens $x_1, x_2, ..., x_t$, a language model estimates:")

st.latex(r"P(x_t \mid x_1, x_2, \ldots, x_{t-1})")

st.markdown("""
*Research: Write 2-3 sentences. Answer — what does this formula mean 
intuitively, what is the model not doing that people assume it is.*
""")

st.divider()

# ─── 2. Markov 1913 ──────────────────────────────────────────────────────────

st.header("2. Where it began — Markov (1913)")

st.markdown("""
*Research: Wikipedia summary of Markov 1913 is sufficient. Write 3-4 sentences.
Answer — what did Markov do, what did he find, why does sequential 
dependence matter, what does this tell us about language.*
""")

st.markdown("""
> Markov, A.A. (1913). *An Example of Statistical Investigation of the Text 
> Eugene Onegin Concerning the Connection of Samples in Chains.*
""")

st.divider()

# ─── 3. Shannon 1948 ─────────────────────────────────────────────────────────

st.header("3. Measuring information — Shannon (1948)")

st.markdown("""
*Research: Shannon 1948 introduction. Write 2-3 sentences.
Answer — what is entropy, why does measuring uncertainty matter 
for language, what did Shannon give us that Markov didn't.*
""")

st.latex(r"H(X) = -\sum_{x} P(x) \log P(x)")

st.markdown("""
*Research: Write 1-2 sentences unpacking the formula.
Answer — what does high entropy mean, what does low entropy mean, 
give a language example.*
""")

st.markdown("""
> Shannon, C.E. (1948). *A Mathematical Theory of Communication.* 
> Bell System Technical Journal.
> [PDF](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
""")

st.divider()

# ─── 4. Shannon 1951 ─────────────────────────────────────────────────────────

st.header("4. Predicting language — Shannon (1951)")

st.markdown("""
*Research: Read Shannon 1951 introduction and conclusion. Write 3-4 sentences.
Answer — what was Shannon's prediction experiment, what did he find,
why is 1 bit per character significant, how does this connect to what 
a language model does.*
""")

st.markdown("Cross entropy — how well does our model's predicted distribution match the true distribution:")

st.latex(r"H(p, q) = -\sum_{x} p(x) \log q(x)")

st.markdown("""
*Research: Write 2-3 sentences.
Answer — what is p, what is q, what does it mean when cross entropy is high,
what does it mean when it approaches true entropy.*
""")

st.markdown("Perplexity — how surprised is the model by the text it sees:")

st.latex(r"PPL = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log P(x_i \mid x_{<i})\right)")

st.markdown("""
*Research: Write 2-3 sentences.
Answer — what does perplexity of 1 mean, what does perplexity of 100 mean,
what is a good perplexity for a language model on English text.*
""")

# ── perplexity widget ─────────────────────────────────────────────────────────

st.subheader("Perplexity — interactive")

prob = st.slider(
    "Average probability assigned to correct token",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01
)

perplexity = 1 / prob

col1, col2 = st.columns(2)
col1.metric("Perplexity", f"{perplexity:.1f}")
col2.metric("Avg probability", f"{prob:.2f}")

st.caption(
    f"A model assigning average probability {prob:.2f} to the correct token "
    f"is as confused as choosing uniformly among {perplexity:.0f} options at every step."
)

st.markdown("""
> Shannon, C.E. (1951). *Prediction and Entropy of Printed English.* 
> Bell System Technical Journal.
> [PDF](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)
""")

st.divider()

# ─── 5. N-gram models ────────────────────────────────────────────────────────

st.header("5. N-gram models — Jelinek & Mercer (1980)")

st.markdown("""
*Research: Any n-gram explainer or Jelinek 1980. Write 2-3 sentences.
Answer — what is an n-gram model, how does it estimate the next token,
why was it practical for speech recognition and machine translation.*
""")

st.latex(r"P(x_t \mid x_1, \ldots, x_{t-1}) \approx P(x_t \mid x_{t-n+1}, \ldots, x_{t-1})")

st.markdown("""
*Research: Write 2-3 sentences on the sparsity problem.
Answer — why does most of the probability space go unseen,
what happens when the model encounters an n-gram it has never seen.*
""")

st.markdown("""
*Research: Write 2-3 sentences on the fixed context problem.
Answer — why is a fixed window a fundamental limitation,
give a concrete example of a long range dependency it would miss.*
""")

# ── n-gram context widget ─────────────────────────────────────────────────────

st.subheader("Context window — interactive")

st.markdown("""
*Write one sentence directing the reader to use the slider below.*
""")

sentence = "The keys that were left on the table by the door were missing"
words = sentence.split()

n = st.slider("N-gram order (n)", min_value=2, max_value=6, value=3)

context = words[-(n):-1]
target = words[-1]
ignored = words[:-(n)]

col1, col2, col3 = st.columns(3)
col1.markdown("**Predicting**")
col1.code(target)
col2.markdown("**Context used**")
col2.code(" ".join(context))
col3.markdown("**Context ignored**")
col3.code(" ".join(ignored) if ignored else "nothing")

st.caption(
    f"A {n}-gram model sees only {n-1} preceding word(s). "
    f"Everything before is invisible regardless of how relevant it is."
)

st.markdown("""
> Jelinek, F. & Mercer, R.L. (1980). *Interpolated Estimation of Markov Source 
> Parameters from Sparse Data.* Pattern Recognition in Practice.
""")

st.divider()

# ─── 6. Neural language models ───────────────────────────────────────────────

st.header("6. Neural language models — Bengio et al. (2003)")

st.markdown("""
*Research: Read Bengio 2003 sections 1 and 2. Write 2-3 sentences.
Answer — what is the key idea, how do embeddings solve the sparsity problem,
why does similarity in embedding space help generalisation.*
""")

st.markdown("""
*Research: Write 2-3 sentences.
Answer — what does the neural language model learn that n-grams cannot,
what is the architecture at a high level, why is it a step change.*
""")

st.markdown("""
> Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). 
> *A Neural Probabilistic Language Model.* 
> Journal of Machine Learning Research, 3, 1137–1155.
> [PDF](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
""")

st.divider()

# ─── 7. The problem that remained ────────────────────────────────────────────

st.header("7. The problem that remained")

st.markdown("""
*Research: No paper needed — reason from what you know. Write 2-3 sentences.
Answer — what fundamental limitation persisted even in neural language models,
why does a fixed context window matter even when embeddings are rich.*
""")

st.markdown("""
*Research: Write 2-3 sentences on RNNs.
Answer — what did RNNs attempt, why did they fail in practice,
what is the vanishing gradient problem in one sentence.*
""")

st.markdown("""
*Research: Write 2-3 sentences — the cliff hanger.
Answer — what mechanism would solve long range dependencies,
what would it need to be able to do that n-grams and RNNs could not.*
""")

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 5])
col1.page_link("pages/foundation/02_tokenization.py", label="Next: Tokenization →")

st.markdown("""
**Further reading**
- Shannon (1948) — [A Mathematical Theory of Communication](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- Shannon (1951) — [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)
- Jelinek & Mercer (1980) — Interpolated Estimation of Markov Source Parameters
- Bengio et al. (2003) — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
""")