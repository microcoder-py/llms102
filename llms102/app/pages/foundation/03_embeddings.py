# app/pages/foundation/03_embeddings.py

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Embeddings | llms102", layout="wide")

# ─── synthetic embeddings ─────────────────────────────────────────────────────
# run once locally to generate artifact
# pip install gensim


@st.cache_data
def load_embeddings():
    vectors = np.load("llms102/artifacts/embeddings/vectors.npy")
    words = np.load("llms102/artifacts/embeddings/words.npy", allow_pickle=True).tolist()
    return dict(zip(words, vectors))

word2vec = load_embeddings()
EMBED_DIM = list(word2vec.values())[0].shape[0]
sorted_words = sorted(word2vec.keys())

# ─── helpers ──────────────────────────────────────────────────────────────────

def get_embedding(word):
    return word2vec.get(word.lower())

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pca_2d(vecs):
    vecs = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:2]]
    return vecs @ components




# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Embeddings")
st.caption("From discrete to continous numerical space")

st.markdown("""
With tokenization, we split up the language space into discrete chunks that we want to perform mathematical operations on, and build a language model. However, they only provide a convenient way to represent tokens in a discrete space - we still don't have a clean way to actually process them. The token IDs in and of themselves possess no information, they are not ordinal in nature, they are only representative tags. 
            
The answer to that, is embeddings. Embeddings project the tokens into a continous space, which is where each token receives a unique position in a vector space such that the interactions between each vector are encoded to help us exploit the statistical relations. 

Imagine if you joined Hogwarts, and were assigned an admission number (i.e. token ID). Currently, you have no information about yourself other than the fact you are a prospective student in a sea of several. Out comes the sorting hat, places you in the house you desire and deserve, which makes the sorting hat the embedding layer. Suddenly, your token ID has meaning. It determines what your interactions with fellow students will be, what you will study, how you comport yourself, etc. The presence of this rich information models how you live through school, which is what we are truly looking for. Do remember, the wand is only to be used in public life under the most serious of circumstances - that holds true for all of you. 
            
> As a playful side note, a single forward pass in this neural network (Hogwarts) takes 7 years to execute, so they implemented parallel execution (different years study at the same time) and group query attention (classrooms share common knowledge with more than one person at the same time) to improve efficiency. It really is wizardry! Voldemort in this case would simply be a bad batch of data, causing training instability, but the Dumbledore optimizer was more than prepared to smoothen out the training curve. I dare say the horcruxes would then be tokenization artifacts arising from poor morphological segmentation of the name 'Voldemort', as we discussed in the last chapter. 
""")  

st.divider()

# ─── 1. What are embeddings? ─────────────────────────────────────────────────

st.header("The history of embeddings")

st.markdown("""
In their 2003 paper entitled [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), Bengio et. al built what is widely recognized as one of the earliest and succesful neural network based language model. These were days yonder, and the language models being studied at this time were either rule based or statistical, often written by experts in their respective fields. Neural networks had begun showing promise in greater capacity for generalizing, and were being explored. 
            
The key issue that had to be tackled was **the curse of dimensionality** - a word sequence that the model would be tested on would likely be radically different from the training set. The paper discusses learning a distributed representation for words instead to provide a common space for interaction, which today we refer to as embeddings.  
            
Using one-hot vectors would likely be wasteful and sparse. For instance, in a one hot setting, the words `cat` and `dog` would have the same distance as `cat` and `justice`. Creating an arithmetic for difference or similarity is a challenge. Therefore a continous learned vector space of lower dimensions made more sense. 

Back then, the atomic unit of tokenization was not a subword token, it was much more common to use entire words. At 100k words (i.e. vocabulary size), for even a sequence of 10 words, there would be `100,000 ^ 10 - 1` or `10 ^ 50 -1` free parameters. Without embeddings, this space would be too large and sparse, disallowing generalization. The authors note this is a common approach used in information retrieval models, but was not directly applied to language modelling. 
            
> To the best of my knowledge, this was the first paper, or at least the most prominent one to use embeddings. Do share any other information you might have with me, feedback is always welcome! 
""")  

st.divider()

# ─── 2. The embedding table ──────────────────────────────────────────────────

st.header("What embeddings are and how they work: Word2Vec")

st.markdown("""
One of the cleanest and best ways to explain embeddings is to walk through Word2Vec. Mind you, it is not the only implementation or the earliest, but it provides us with a complete picture. 
            
The core logic is simple. If two words frequently reoccur in a sequence, they are more related than two words that rarely appear together. Consider a sentence, `The cat sat on the mat`. If in the training corpora, we see cat and sat together a lot of times, we can attach semantic meaning to cats (which in this case would be that they are lazy, and that is not untrue). Additionally, let's assume the sentences `The cat hissed at the dog` and `The cat slept on the shelf` also appear. This way, the word `cat` is associated often with `dog` and `sleep` and `sat`, creating a cross-relation between those words as well. 
            
How do we exploit this statistical nature? Let us discuss the skip-gram method used for Word2Vec (the other is Continuous Bag of Words, very similar in nature of intuition). For `the cat sat on the mat`, we construct co-occuring words by considering a context window of fixed size, looking to all tokens to left and right. This matters because words that appear closer in sentences are likely more correlated than those further apart. This is a design choice, and we will discuss other choices later as well.           
""")  

st.markdown("""
```
centre word        context window (size=2)
─────────────────────────────────────────
     the      →   [cat]
     cat       →   [the, sat]
    [sat]      →    the, cat, on, mat      ← focus word
     on        →   [sat, the]
     the       →   [on, mat]
     mat       →   [the]

loss = -log P(the|sat) - log P(cat|sat) - log P(on|sat) - log P(mat|sat)
```
""")

st.markdown("""
We use an embedding table, find the embedding, and use a single layer to predict what the surrounding words might be. This way, the embedding table learns what words co-occur frequently, so the learned representation highlights that cleanly. 
            
Overall, the process would look like
            
| Step | Operation | Example |
|------|-----------|---------|
| 1 | Take centre word | `sat` |
| 2 | Look up embedding | `sat` → `[0.2, -0.5, 0.8, ...]` |
| 3 | Predict context words | `the`, `cat`, `on`, `mat` |
| 4 | Compare predictions to actual context | predicted `cat` with prob `0.6`, actual `cat` |
| 5 | Compute cross entropy loss | $-\\log(0.6) = 0.51$ |
| 6 | Backpropagate | update embedding for `sat` |
| 7 | Repeat for all centre words | embedding table converges |
""")

st.info("The actual implementation takes other factors into consideration, we are only building intuition here. The interested reader is encouraged to look into the paper")

st.subheader("Visualising Embeddings")

st.markdown("""
As we mentioned, embeddings are in effect vectors that encode semantic information. These dimensions are not directly interpretable, they encode information based on training methodology and data, but the relations are captured accurately. Let's first visualise the embeddings of some common words using Word2Vec
""")

token_input = st.selectbox(
    "Select a word",
    options=sorted(word2vec.keys()),
    index=sorted(word2vec.keys()).index("king"),
    key="context_word"
)

if token_input:
    embedding = get_embedding(token_input)

    col1, col2 = st.columns([1, 3])
    col1.metric("Word", token_input)
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

st.caption(
    "Embeddings generated using sentence-transformers/all-MiniLM-L6-v2 for demonstration. "
    "Real Word2Vec embeddings follow the same structure."
)

st.markdown("""
Watching these values, however, provides us only the representation. As always, we ask, what do the numbers mean? 
            
The interesting part about embeddings is that because they encode the underlying relations between words, we can actually perform arithmetic on them to compare and contrast them. 
""")

st.subheader("Do the embeddings encode any information?")

st.markdown("""
As a simple test, let's project the vectors into 2D space using PCA. We can see very clear clusters appear for each, highlighting how the embeddings push similar terms closer to each other. 
""")

categories = {
    "royalty":   ["king", "queen"],
    "gender":    ["man", "woman"],
    "geography": ["paris", "london", "france", "england"],
    "animals":   ["cat", "dog", "fish", "bird"],
    "sentiment": ["good", "bad", "happy", "sad"],
    "nature":    ["water", "fire", "earth", "wind"],
    "other":     ["hello", "world", "fast", "slow"],
}

word_to_category = {
    word: category
    for category, words_list in categories.items()
    for word in words_list
}

sample_words = [w for w in word_to_category if w in word2vec]
vecs = np.array([word2vec[w] for w in sample_words])
colors = [word_to_category[w] for w in sample_words]

reduced = pca_2d(vecs)

fig = px.scatter(
    x=reduced[:, 0],
    y=reduced[:, 1],
    text=sample_words,
    color=colors,
    title="Word2Vec embeddings projected to 2D (PCA)",
    labels={"x": "PC1", "y": "PC2", "color": "Category"}
)
fig.update_traces(textposition="top center", marker=dict(size=10))
fig.update_layout(showlegend=True)
st.plotly_chart(fig, use_container_width=True)

PAIRS = {
    "Royalty & Gender: king − man vs queen − woman": (("king", "man"), ("queen", "woman")),
    "Country & Capital: france − paris vs england − london": (("france", "paris"), ("england", "london")),
    "Animal pairs: cat − dog vs fish − bird": (("cat", "dog"), ("fish", "bird")),
    "Sentiment: good − happy vs bad − sad": (("good", "happy"), ("bad", "sad")),
    "Speed: fast − slow vs good − bad": (("fast", "slow"), ("good", "bad")),
}


st.markdown("Additionally, since these are vectors, common operations such as directionality, addition hold. Do not that this is an approximation of a vector space, and not an exact one. It does not form a valid vector space in the mathematical sense, but we can see that the sense of directionality and similarity hold - crucial operations for language modelling") 

st.markdown("""
For instance, the vector directed between `king` and `man` should ideally have similar direction and magnitude as the one between `queen` and `woman`, since the gender is the core separating factor here.
            
Of course, the actual values depend heavily on how well the relations were encoded, which is a model specific detail, but in general the principle holds. 
""")

pair_choice = st.selectbox("Choose a pair", options=list(PAIRS.keys()))

(word_a1, word_a2), (word_b1, word_b2) = PAIRS[pair_choice]

vec_a = get_embedding(word_a1) - get_embedding(word_a2)
vec_b = get_embedding(word_b1) - get_embedding(word_b2)

similarity = cosine_similarity(vec_a, vec_b)

col1, col2, col3 = st.columns(3)
col1.metric(f"{word_a1} − {word_a2}", "direction A")
col2.metric(f"{word_b1} − {word_b2}", "direction B")
col3.metric("Cosine similarity", f"{similarity:.3f}")

st.caption(
    f"If embeddings encode this relationship consistently, both directions "
    f"should point the same way in vector space. "
    f"Similarity of {similarity:.3f} {'supports this' if similarity > 0.5 else 'suggests the relationship is not strongly encoded'}."
)
st.divider()

# ─── 7. What embeddings are not ──────────────────────────────────────────────

st.header("What embeddings are not")

st.markdown("""
While embeddings are a very useful tool, and help us encode inter-token relationships very well, it helps to understand what they are not 
            
1. **Embeddings are not meaning** - As mentioned, the embeddings depend quite heavily on training method and data used. They do not capture absolute information, only relative information based on the training. This shows us why it is important to have good tokenization and sufficient representation samples for each token. It is also why rarely occuring tokens tend to emit poorer performance from the model - it's because the encoded representation is not strong enough to truly highlight underlying semantics. 
            
2. **Embeddings are not fixed** - We will see later that embeddings provide us a neat way to capture interactions between tokens, but this does not mean they must remain static. Embeddings are modified in each layer of the transformer (which we will see later), and can also be updated with fresh information via finetuning. For now, only consider them to be a tool in the arsenal, a great way to represent discrete tokens in a continuous space. The examples we provided above showed them to be static, but that is because Word2Vec is static by design. This is not true of all models. 
            
3. **Embeddings are not the whole story** - Embeddings carry information, but there is a lot more to be done with said information. We must combine it, process it, extract inter-token dependencies, and a lot more, which we do with transformer networks. 
""")  

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