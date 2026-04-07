import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.code_loader import show_source, run_it_yourself
st.set_page_config(page_title="Loss and Perplexity | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Loss and Perplexity")
st.caption("What is the model minimizing?")

st.markdown("""
In the [language modelling chapter](language_modelling) we have already seen and understood the most basic loss function for language modelling. Here, I just want to provide a surface level intuition for what we are actually looking at when we try and make our models follow instructions, or reason. 
            
Recall that language modelling (for causal language modelling) is basically finding the distribution for this formula
""")

st.latex(r"P(x_t \mid x_1, x_2, \ldots, x_{t-1})")

with st.expander("Formula Intuition for loss calculation", expanded=False):
    st.markdown(r"""
    When we calculate loss for LM, we use the LM Head of the transformer to project a $Sequence Length, Vocabulary Size$ tensor. For each token in the sequence, we have a one-hot encoding of the target variable. For instance, if our vocab size was 4, we would have for each token a set of probs of vocab size, say $[0.3, 0.4, 0.1, 0.2]$ (this is our $p(x)$) while the target would only contain the correct token, let's say it's token 4, so the target labels would be $[0, 0, 0, 1]$ (which is our $q(x)$). 
                
    The estimated probabilities are what our model thinks the resulting distribution should be like, while the one-hot encoded labels hold the ground truth. Using our formulae, we can then try and map the two distributions such that $p(x) \approx q(x)$ as well as it can
                
    Where are the conditional probabilities coming from in this case? Our formula depends on the past tokens in the sequence, but we cannot see it explicitly in the loss calculation. Well, remember that we abstracted the whole LLM as a single mathematical function, and that within this black box we have attention networks, and other components and intermix information between tokens? That helps create the conditionality. These output probabilities are directly dependent on the past tokens in the sequence.
    """)

st.header("Maximum Likelihood Estimation")
st.markdown("""
As we mentioned before, to create the perfect language model, we would require all possible language data in the world. That would form the truest distribution possible. However, we do not have access to this distribution, all we have are data points collected from various sources. We approximate the true data distribution from this subset, and try to map our data to it. 
            
For instance, if you had a slightly biased coin that you flipped 10 times, and you want to predict what the likelihood of getting a heads or tails is, all you have access to is the data you have seen. You could consider flipping it another 100 times to receive a more accurate estimate, but wouldn't 1000 tries be even more accurate? As you can see, we would end up requiring infinite samples before making the most accurate inference, which is not feasible for us. Instead, we use our existing data samples, and assume that at some point, it's enough for creating a good distribution.

**Key Takeaway**: We never have access to the true data distribution. We have some subset of observed values, which we use to work backward and estimate the parameters that bring us to the true distribution         
""")

st.subheader("Formalising the intuition")
st.markdown(r"""
This is exactly what **Maximum Likelihood Estimation (MLE)** formalises. Given $n$ observed
data points $x_1, x_2, \ldots, x_n$, we define the **likelihood function** as the probability
of seeing that data under a model parameterised by $\theta$:

$$
L(\theta \mid x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid \theta)
$$

**Back to the coin flip:** if $\theta$ is the probability of heads and we observed 7 heads in
10 flips, then $P(\text{data} \mid \theta) = \theta^7 (1-\theta)^3$. Given this distribution, our question is this - at what $\theta$ (i.e. our parameters) does this distribution give us the maximal value? Maximising this gives $\hat{\theta} = 0.7$ — exactly the frequency we observed. More flips would push this estimate
closer to the true bias, which is the same intuition we had before formalising it.
""")


H, T = 7, 3

def likelihood(theta):
    return np.power(theta, H) * np.power(1 - theta, T)

def log_likelihood(theta):
    return H * np.log(theta) + T * np.log(1 - theta)

thetas = np.linspace(0.01, 0.99, 200)
likelihoods = likelihood(thetas)

theta = st.slider("θ (probability of heads)", min_value=0.01, max_value=0.99,
                  value=0.70, step=0.01, format="%.2f")

lv  = likelihood(theta)
llv = log_likelihood(theta)

col1, col2, col3 = st.columns(3)
col1.metric("θ (your guess)",   f"{theta:.2f}")
col2.metric("L(θ) = θ⁷·(1−θ)³", f"{lv:.6f}")
col3.metric("log L(θ)",          f"{llv:.3f}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=thetas, y=likelihoods,
    mode="lines",
    name="L(θ)",
    line=dict(color="#7F77DD", width=2),
))

fig.add_trace(go.Scatter(
    x=[theta], y=[lv],
    mode="markers",
    name="selected θ",
    marker=dict(color="#1D9E75", size=10),
))

fig.add_trace(go.Scatter(
    x=[0.70], y=[likelihood(0.70)],
    mode="markers",
    name="MLE peak (θ=0.70)",
    marker=dict(color="#D85A30", size=10, symbol="diamond"),
))

fig.update_layout(
    xaxis_title="θ (probability of heads)",
    yaxis_title="L(θ) = θ⁷(1−θ)³",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=40, l=60, r=20),
    height=350,
)

st.plotly_chart(fig, use_container_width=True)

t_str    = f"{theta:.2f}"
comp_str = f"{1 - theta:.2f}"
t_pow    = f"{theta**H:.4f}"
c_pow    = f"{(1-theta)**T:.4f}"
note     = " — the highest possible value" if abs(theta - 0.70) < 0.005 else \
           f" — lower than peak at θ = 0.70" if lv < likelihood(0.70) else ""

st.caption(
    f"L({t_str}) = {t_str}⁷ × {comp_str}³ = {t_pow} × {c_pow} = **{lv:.6f}**{note}"
)

st.markdown(r"""
Now, the derivative of this probability distribution is simply 

$$
\frac{dL}{d\theta} = 7\theta^6(1-\theta)^3 - 3\theta^7(1-\theta)^2
$$            
            
$$
= \theta^6(1-\theta)^2(7 - 10\theta)            
$$
            
We know that the maximal value of this distribution occurs where $\frac{dL}{d\theta} = 0$, giving us 

$$
\theta^6(1-\theta)^2(7 - 10\theta) = 0            
$$ 
            
This equation can only be non-trivially solved for $\theta \in (0, 1)$, therefore the only solution is that $(7 - 10\theta) = 0$ or that $\theta = 0.7$, which is exactly the result we see in our graph as well. 
            
Language modelling requires us to multiply the probabilities of multiple such probabilities across the vocabulary space. Because multiplying many small probabilities leads to numerical underflow, we take the log (which turns products into sums without changing the location of the maximum):

$$
\ell(\theta) = \sum_{i=1}^{n} \log P(x_i \mid \theta)
$$

Our MLE estimate $\hat{\theta}$ is the parameter value that maximises this. It also converts high value exponents (more complex to calculate derivatives for) into simple additions, but preserves the maxima location, since logarithm is a monotonically increasing function

$$
\hat{\theta} = \underset{\theta}{\arg\max} \sum_{i=1}^{n} \log P(x_i \mid \theta)
$$
            
Maximising this, or minimizing the negative of this value are equivalent. Therefore, we define negative log-likelihood. 
            
$$
\text{NLL}(\hat{\theta}) = \underset{\theta}{\arg\min} -\sum_{i=1}^{n} \log P(x_i \mid \theta)
$$

Why does this formula feel familiar? Because it looks exactly the same as [Cross Entropy Loss](language_modelling#6-cross-entropy-and-kl-divergence)! Recall from the [transformer architecture chapter](transformer_block#implementation) that in our implementation we output the logits, and not the softmaxed values. CE Loss effectively first applies softmax alongtoken dimension, followed by Negative Log Likelihood. For NLL, we would have first applied some non-linearity to convert the distribution into a probability distribution, which is achieved via softmax in CE Loss.
            
NLL/CE loss also offer us a nice property for backpropagation. If the difference between actual and estimated probability of output is high, the penalty is high, but it drops off logarithmically for lower values. Notice the marked band where $ q(x) \approx p(x)$. If you're wondering why the loss is not necessarily zero at $q(x) = p(x)$, it's because Cross Entropy Loss does not only measure the mismatch between the two distributions (the Kullbeck-Leibler divergence), it also considers the absolute uncertainty of $p$ (entropy of distribution). Language modelling necessarily demands both terms are considered.
""")
px   = st.slider("p(x) fixed", 0.01, 0.99, 0.7, step=0.01)
rmax = st.slider("ratio r = p/q max", 0.1, 10.0, 5.0, step=0.1)

r_vals = np.linspace(0.01, rmax, 300)

# Cross-entropy
ce = px * np.log(r_vals) - px * np.log(px)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=r_vals,
        y=ce,
        name="CE loss",
        line=dict(color="#1D9E75", width=2)
    )
)

# --- Highlight region near r = 1 ---
r_low = 0.8
r_high = 1.2

fig.add_vline(x=1.0, line=dict(color="white", dash="solid", width=1))
fig.add_vline(x=r_low, line=dict(color="#888", dash="dot"))
fig.add_vline(x=r_high, line=dict(color="#888", dash="dot"))

# Optional shaded region
fig.add_vrect(
    x0=r_low, x1=r_high,
    fillcolor="gray", opacity=0.15,
    line_width=0
)

fig.update_layout(
    xaxis_title="r = p(x) / q(x)",
    yaxis_title="Cross-Entropy Loss",
    margin=dict(t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.header("\'Reading\' Cross-Entropy Losses")

st.markdown(r"""
CE Loss actually gives us a very convenient way to directly observe how well or or how poorly our model is performing. Recall that perplexity is simply $e^{\text{ce\_loss}}$ (for simplicity, we use base $e$ in CE Loss calculations)

What does perplexity indicate? In the multilabel classification problem, which is what we deal with in language modelling (we need to predict the likelihood probabilities between multiple possible tokens at each step), perplexity tells us how uncertain the model is about choosing a given token. If we had a perplexity of 50, the model on average is struggling to decide which of the 50 vocabulary tokens to select, i.e. it effectively assigns the same max likelihood to fifty different vocabulary tokens. 

How does this relate to CE Loss? Given that 

$$
\text{PPL} = e^{\text{CE Loss}}
$$

Let's plot CE Loss against perplexity. For a completely random model, we would assign equal probabilities to each token class, so for 32000 vocabulary size, each token would always receive a likelihood of $\frac{1}{32000}$, which means the CE Loss would be $-\ln\left(\frac{1}{32000}\right)$ or $10.3734$. At any value higher than this loss, your model is effectively performing worse than even a random model for this vocabulary size. 
""")

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# CE range (keep reasonable)
ce_max = 12
ce_vals = np.linspace(0.0, ce_max, 400)
ppl_vals = np.exp(ce_vals)

fig = go.Figure()

# Main curve
fig.add_trace(
    go.Scatter(
        x=ce_vals,
        y=ppl_vals,
        name="Perplexity",
        line=dict(width=3)
    )
)

# Powers of 2 vocab sizes
vocab_sizes = [2**i for i in range(10, 18)]  # 1024 → 131072

for vocab in vocab_sizes:
    ce_random = np.log(vocab)

    # Vertical line
    fig.add_vline(
        x=ce_random,
        line=dict(color="#FF8C00", dash="dot", width=1.5),
        opacity=0.8
    )

    # ✅ Small label near x-axis
    fig.add_annotation(
        x=ce_random,
        y=vocab,
        text=f"{vocab}",
        showarrow=False,
        textangle=-90,
        xanchor="center",
        yanchor="bottom",
        yshift=12,
        font=dict(size=10, color="#FF8C00")
    )

# Dummy trace for legend
fig.add_trace(
    go.Scatter(
        x=[ce_vals[0], ce_vals[1]],
        y=[ppl_vals[0], ppl_vals[1]],
        mode="lines",
        line=dict(color="#FF8C00", dash="dot", width=2),
        name="CE loss for fully random models (varying vocab size)"
    )
)

# Layout
fig.update_layout(
    xaxis_title="Cross-Entropy Loss",
    yaxis_title="Perplexity",
    margin=dict(t=40, b=40),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
Recall that CE Loss works on a logarithmic scale, therefore the gap between CE Losses for different vocabulary sizes is actually quite small. A random model with 128k vocabulary size would show an average CE Loss of $-\ln\left(\frac{1}{128000}\right)$ or $11.7598$. So if you ever see your model hitting losses of some magnitude, you can start thinking in terms of how many tokens it is perplexed about, and whether or not it makes sense given the stage of training. 
            
In particular, if you ever see CE Loss values higher than 13, your model cannot distinguish between more than 440k tokens - which means you aren't modelling the distribution at all. At 13.8, that extends to being unable to distinguish between 1 million possibilities. It might make more sense to switch off the training entirely and begin looking into why the perplexities are so high - there may be a bug in the code, or perhaps parameters. I once encountered a CE Loss of 112 while performing quantization aware training on a billion parameter model, and still let the training run. The model, unsurpisingly, never converged, and I had by that point burned through around 5 USD on GPU costs. Sure, not a huge amount, but that is besides the point. There was a tiny bracket error in my code, which should have been apparent the moment I saw such a massive spike in CE Loss values. Don't be like me. Learn from my mistakes.              
""")

st.divider()

st.markdown("""
As we begin exploring further, diving deeper into how losses are calculated for different methods and stages (Base model, Instruction Following, Complex Reasoning, Multistep-Reasoning etc.) you will notice we use different types of loss functions. They will be discussed as and when we approach them. 
            
The key takeaway for now, is that ultimately what we are trying to do is map one distribution to another. The specific ways in which we do this is method and stage variant, but the end goal remains the same - how can this model begin sounding closer to the target distribution? Cross Entropy loss sits at the center of this discussion, since this is where base model tuning happens, which is why an entire chapter was dedicated to developing this intuition.
""")

col1, col2 = st.columns([3, 5])
col1.page_link("pages/training/11_lr_schedules.py", label="Next: Learning Rate Schedules →") 