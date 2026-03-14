# app/pages/foundation/01_language_modelling.py

import streamlit as st
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Language Modelling | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Language Modelling")
st.caption("llms102 — the last lesson you will need")

st.markdown("""
Languages are not random. We cannot throw in words in any sequence and expect them to make sense. For instance, *The dog ate the cat's food* is a valid sentence. However, *cat eat dog the food* contains all the same words and conveys no meaning. 

The main point here, is that language is based on a set of rules - or statistical properties. This means that we can exploit said statistical property to perform tasks on the language space. 
""")

st.divider()

# ─── 1. What is a language model? ────────────────────────────────────────────

st.header("1. What is a language model?")

st.markdown("""

The simplest way to imagine this is to think of words as a long chain, with each next word being heavily dependent on the past words, in effect a probability distribution. That is a language model. 
""")


st.markdown("Considering that most models today primarily target causal language modelling, we focus most on this. Bear in mind, there are other types, such as conditional language models, which we will touch on in further sections")

st.markdown("Formally, given a sequence of tokens $x_1, x_2, ..., x_t$, a causal language model estimates:")

st.latex(r"P(x_t \mid x_1, x_2, \ldots, x_{t-1})")

st.markdown("""
As we mentioned in the example before, this model in essence sees all existing tokens, and decides what next token would fit best. The precise underlying mechanism for developing the probability distribution is our LLM, which is the focus of AI research on many fronts. 
            
For now, focus less on what the function is. What we first need to do is create an intuition for what language modelling itself is. We will study some of its history, and understand how it works. 
""")

st.divider()

# ─── 2. Markov 1913 ──────────────────────────────────────────────────────────

st.header("2. Where it began — Markov (1913)")

st.markdown("""
The formal mathematical study of sequential dependence in language began not with a linguist, but with a mathematician trying to prove a point about probability theory.

In 1913, Andrey Markov manually analysed 20,000 characters from Pushkin's novel *Eugene Onegin*, counting sequences of vowels and consonants by hand. What he found was that each letter was not independent — the probability of a vowel or consonant depended heavily on what came before it. Language, at least at the character level, had memory.

This idea — that the next symbol in a sequence depends on preceding symbols — is called the **Markov property**, and it is the statistical foundation every language model is built on, including the transformers we will build in this series.
""")

st.markdown("""
> Markov, A.A. (1913). *An Example of Statistical Investigation of the Text 
> Eugene Onegin Concerning the Connection of Samples in Chains.*
""")

st.divider()

st.header("3. Zipf's Law (1935)")

st.markdown("""
Before we can measure language statistically, it helps to first observe that language *has* measurable statistical structure at all. As an example, we explore Zipf's law. 

In 1935, linguist George Zipf observed that in any large corpus of text, the most frequent word appears roughly twice as often as the second most frequent, three times as often as the third, and so on. Rank a language's vocabulary by frequency, and frequency falls as a power law of rank.

This holds across every language ever studied — English, Mandarin, Arabic, ancient Latin. Language is not random. It is deeply, predictably structured.
            
Zipf was not the first to find this power law, but it is still called the Zipf law. In fact, as a linguist, he had a disdain for math, which he did mention in writing. 
""")

# ── Zipf widget ───────────────────────────────────────────────────────────────

ranks = np.arange(1, 500)
frequencies = 1 / ranks

fig = px.line(
    x=np.log(ranks),
    y=np.log(frequencies),
    labels={"x": "log(rank)", "y": "log(frequency)"},
    title="Zipf's Law — word frequency vs rank (log-log scale)"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Zipf, G.K. (1935). The Psycho-Biology of Language.")

# ─── 3. Shannon 1948 ─────────────────────────────────────────────────────────

st.header("4. Measuring information — Shannon (1948)")

st.markdown("""
If I were to highlight two scientists I aspire to, they would be Alan Turing and Claude Shannon. Shannon in particular was a mathematical genius who focused more on intuition than exact formulation, often needing assistance from others to help translate his brilliance into equations. 
            
One of the most influential scientists of the 20th century, he is best known for creating **Information Theory**, the field of study focused on understanding the information content of a probability distribution. This allowed us to create the digital era, being able to encode, transmit and correct data being sent over very noisy channels back in the day. 

A fun side note - his original paper was written in such simple language and formulation that scientists split up into two factions - one who acknowledged his genius, the other unwilling to believe something so simple could have such outlandish consequence. 
""")

st.divider()

st.markdown("""
The core idea behind information theory is quite simple - the lower the likelihood of an event's occurence, the more information it provides. 
            
For example, if you had a guard dog who barked each night at every passing leaf, you wouldn't care if the dog barked tonight. However, if the dog is always silent, and suddenly barks tonight, you would know there is cause for concern. 
            
This inverse relation between how informative an event is and its statistical likelihood form the basis of information theory 

Based on this understanding, and a lot of mathematical wrangling, we arrive at a formula for the information content of an event
""")

st.latex(r"I(x) = -\log P(x)") 

st.markdown("""
When averaged over all events, it gives us the average information content of the distribution, otherwise referred to as **Shannon Entropy**
""")

st.latex(r"H(X) = -\sum_{x} P(x) \log P(x)")

st.markdown("""
If the entropy of a distribution is very high, it tells us that the events are relatively random, with higher information content per event 
            
However, if the entropy is low, it tells us that our probability distribution contains lesser information, i.e. is easier to predict and model  
            
This intuition is critical to develop, since entropy is what we use for almost all language modelling - our key intent is to find the probability distribution that matches as closely as possible to natural language. 
""")

st.markdown("""
From entropy we derive **Perplexity** — a more intuitive measure of how surprised the model is by the text it sees. Note that here we use natural log, so perplexity is expressed in nats rather than bits.
""")

st.latex(r"PPL = e^{H(p,q)}")

st.markdown("""
A perplexity of 2 means the model is choosing between 2 equally likely options at every step. A perplexity of 100 means it is as lost as a random guess among 100 options. Lower is better.
""")

# ── perplexity widget ─────────────────────────────────────────────────────────

st.subheader("Perplexity — interactive")

prob = st.slider(
    "Average probability assigned to correct token. Calculated as 1/p, since entropy calculation inverts the base of the logarithm. In the case of language modelling, we always use base e",
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
    f"Assigning average probability {prob:.2f} to the correct token "
    f"is equivalent to choosing uniformly among {perplexity:.0f} options at every step."
)

st.markdown("""
> Shannon, C.E. (1948). *A Mathematical Theory of Communication.* 
> Bell System Technical Journal.
> [PDF](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
""")

st.divider()

# ─── 4. Shannon 1951 ─────────────────────────────────────────────────────────

st.header("5. Predicting language — Shannon (1951)")

st.markdown("""
In his 1951 paper, Shannon asked subjects to guess the next character in a sequence one at a time, given all preceding characters.

Why bother with this? We had already established that language has statistical structure. This paper empirically verified it. Shannon argued that speakers of any language internalise grammar and syntax rules, which amounts to a statistical model — the question was how much certainty that model provides.

Theoretically, with 27 possible characters, the maximum entropy would be:
""")

st.latex(r"\log_2(27) \approx 4.75 \text{ bits per character}")

st.markdown("""
However, based on his experiments, the estimated uncertainty was only **1 bit per character** — indicating that at the character level, English is almost 80% predictable.

This finding is significant for language modelling — it gives us a lower bound on perplexity:
""")

st.latex(r"2^1 = 2")

st.markdown("""
Meaning a perfect language model would almost never be surprised by the next character. Modern LLMs with subword tokenization already achieve single digit perplexities, coming remarkably close to Shannon's theoretical limit.
""")

st.header("6. Cross Entropy & KL Divergence")

st.markdown("""
Shannon gave us a way to measure the intrinsic uncertainty of a distribution. But in language modelling, we have two distributions to contend with — the true distribution of language, and our model's approximation of it.

The true distribution $p$ represents natural language as it actually is — the real probabilities of every possible sequence. We never have direct access to this. What we have is text, which are samples drawn from $p$.

Our model produces a distribution $q$ — its best guess at $p$ given what it has learned. The question becomes: how do we measure how wrong $q$ is?
""")

st.markdown("**Cross entropy** measures the average number of bits needed to encode events from $p$ using a code optimised for $q$:")

st.latex(r"H(p, q) = -\sum_{x} p(x) \log q(x)")

st.markdown("""
When our model is perfect and $q = p$, cross entropy equals Shannon entropy — we are encoding language as efficiently as possible. When our model is wrong, cross entropy is higher. The difference between the two is the **KL Divergence**:
""")

st.latex(r"D_{KL}(p \| q) = H(p, q) - H(p)")

st.markdown("""
KL divergence measures the information lost by using $q$ instead of $p$. It is always non-negative — you can never do better than the true distribution — and equals zero only when $q = p$ exactly.

This gives us a clean picture of what language modelling actually is:
""")

col1, col2, col3 = st.columns(3)
col1, col2, col3 = st.columns(3)

col1.markdown("**Shannon Entropy H(p)**")
col1.markdown("Intrinsic uncertainty of language — the theoretical lower bound we cannot beat.")

col2.markdown("**Cross Entropy H(p,q)**")
col2.markdown("What our model actually achieves — always ≥ H(p).")

col3.markdown("**KL Divergence**")
col3.markdown("The gap between our model and reality — what training tries to close.")

st.markdown("""
Training a language model is simply minimising cross entropy over text samples. Since $H(p)$ is fixed — we cannot change the intrinsic uncertainty of language — minimising cross entropy is equivalent to minimising KL divergence, pulling our model's distribution as close as possible to the true distribution of natural language.
""")

st.divider()

st.markdown("""
Language is statistical. That is all we need to know to begin.

What follows is the story of building a parametric function $f_\\theta$ that approximates the true distribution of language $p$ as closely as possible:
""")

st.latex(r"q_\theta(x_t \mid x_{<t}) \approx p(x_t \mid x_{<t})")

st.markdown("""
Information theory gives us the tools to measure how well we are doing — entropy tells us the theoretical limit, cross entropy tells us where we are, and KL divergence tells us how far we have to go.

The rest is just finding $\\theta$, which happens to be our LLM.  
            
Of course, as we dive deeper into the course, we will also see other methods of updating parameters to better suit our needs, but it's only a question of _how_ we want $\\theta$ to operate.
""")

st.divider()

st.markdown("""
**Further reading**
- Shannon (1948) — [A Mathematical Theory of Communication](http://math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- Shannon (1951) — [Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)
- Jelinek & Mercer (1980) — Interpolated Estimation of Markov Source Parameters
- Bengio et al. (2003) — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
""")