import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import pandas as pd

from utils.code_loader import show_source, run_it_yourself
st.set_page_config(page_title="Dynamic Batching and Sequence Packing | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Dynamic Batching and Sequence Packing")
st.caption("When data meets GPUs")

st.markdown("""
When we crawl through the internet for data, we tend to have data of varying lengths. However, when we train a model, within each batch, we expect all input tensors to be of the same size. The reasoning is simple, GPUs need matrices, not random data, so they must be filled to the fullest extent so that each batch appears as a single matrix. 
            
We use special tokens to denote specific information to the model. As these special tokens reappear in position, the model learns how to understand their meaning. For our case, we need to look at padding and end of sequence tokens in particular
""")


st.header("Padding and End Of Sequence (EOS) Tokens") 

st.markdown("""
We introduce two tokens, Padding and EOS. EOS, as the name states, shows the model that this is the end of the current sequence. With enough samples, the model ultimately learns sequence boundaries. 
            
Padding token is more so a zero information token - basically used to highlight that there is no need to focus on this particular token. It's sort of like write 100 as 000100 to make it fit in with six digit numbers. The initial three zeros do not change the underlying numerical representation. 
            
Each tokenizer uses different styles of both EOS and Padding tokens, make sure to check your own tokenizers. Very often though, we use EOS as the padding token. 
""")

st.divider()

st.header("Issues with variable length data")

st.markdown("""
Let's say we have a brutally simple dataset of only a few sentences that we will train our model on. For convenience, our pad token will be `<PAD>` and EOS token will be `<EOS>`

- The dog runs fast. 
- Deep learning is fun. 
- I like tea. 
- Tokenization helps models.

### Subword token alignment (no padding)

Let's assu

| Sentence   | T1     | T2     | T3      | T4      | T5      | T6      | T7      | T8      |
|------------|--------|--------|---------|---------|---------|---------|---------|---------|
| Sentence 1 | The    | dog    | runs    | fast    | .       |         |         |         |
| Sentence 2 | Deep   | learn  | ##ing   | is      | fun     | .       |         |         |
| Sentence 3 | I      | like   | tea     | .       |         |         |         |         |
| Sentence 4 | Token  | ##iza  | ##tion  | helps   | model   | ##s     | .       |         |

Empty cells indicate that the sentence has ended (no padding applied). Even with short sentences, subword tokenization introduces variability in token counts across sequences. This cannot be processed by an LLM directly since this isn't a complete matrix. First, we will add EOS tokens, and then pad the sequences to the same lengths. 

| Sentence   | T1     | T2     | T3      | T4      | T5      | T6      | T7      | T8      |
|------------|--------|--------|---------|---------|---------|---------|---------|---------|
| Sentence 1 | The    | dog    | runs    | fast    | .       | <EOS>   | <PAD>   | <PAD>   |
| Sentence 2 | Deep   | learn  | ##ing   | is      | fun     | .       | <EOS>   | <PAD>   |
| Sentence 3 | I      | like   | tea     | .       | <EOS>   | <PAD>   | <PAD>   | <PAD>   |
| Sentence 4 | Token  | ##iza  | ##tion  | helps   | model   | ##s     | .       | <EOS>   | 
            
Notice how shorter sequences have a ton of `<PAD>` tokens because we are trying to map them to the longest possible sequence in the batch. This is incredibly wasteful, since we now have to process these zero information tokens with full GPU usage. Let's visualize how many `<PAD>` tokens might appear. Obviously, the closer min and max sequence lengths are, the more efficient the pipeline
""")

st.markdown("""
Simulate how much padding is introduced when batching variable-length sequences.

**Scenario:** 1024 sequences, max length = 1024 tokens.
""")
num_sequences = st.slider("Number of sequences", 64, 2048, 1024, step=64)
max_length = st.slider("Maximum sequence length", 128, 2048, 1024, step=64)
min_length = st.slider("Minimum sequence length", 1, max_length, 128)

# Generate random lengths
lengths = [random.randint(min_length, max_length) for _ in range(num_sequences)]
batch_max = max(lengths)

# Compute padding
pad_tokens = [batch_max - l for l in lengths]

total_tokens = batch_max * num_sequences
real_tokens = sum(lengths)
total_pad = sum(pad_tokens)
utilization = real_tokens / total_tokens

# Statistics table
st.subheader("Batch Statistics")

stats_df = pd.DataFrame({
    "Metric": [
        "Number of sequences",
        "Batch max length",
        "Total real tokens",
        "Total padded tokens",
        "Padding percentage",
        "Token utilization"
    ],
    "Value": [
        str(num_sequences),
        str(batch_max),
        str(real_tokens),
        str(total_pad),
        f"{100 * total_pad / total_tokens:.2f}%",
        f"{100 * utilization:.2f}%"
    ]
})

st.table(stats_df)

st.markdown("""
**Visual intuition**: We have compacted the representation of up to 1024 tokens per sequence into just 16 bins, i.e. each bin represents **64** tokens, making it easy to see how much of each sequence is wasted on `<PAD>` tokens.
""")
sample_n = min(8, num_sequences)

# We compress along sequence length (not number of sequences)
num_bins = 16
bin_size = max(1, batch_max // num_bins)

rows = []
for i in range(sample_n):
    l = lengths[i]
    p = pad_tokens[i]

    real_bins = int(l // bin_size)
    pad_bins = num_bins - real_bins

    real_bar = "🟦" * real_bins
    pad_bar = "⬜" * pad_bins

    rows.append({
        "Sequence": i + 1,
        "Length": l,
        "Padding": p,
        "Visualization": real_bar + pad_bar
    })

viz_df = pd.DataFrame(rows)

st.dataframe(viz_df, hide_index=True)

st.header("Dynamic Batching by Length")

st.markdown("""
One very simple solution to the above issues is to sort the sequences by length, and then batch them. This way, each batch would contain similarly sized sequences, with different batches being differently sized. As an example, let's run the same simulation as above for given batch size which you can control and explore 
""")

# Controls
dyn_num_seq = st.slider("Number of sequences", 64, 2048, 1024, step=64, key="dyn_num_seq_v2")
dyn_max_len = st.slider("Maximum sequence length", 128, 2048, 1024, step=64, key="dyn_max_len_v2")
dyn_min_len = st.slider("Minimum sequence length", 1, dyn_max_len, 128, key="dyn_min_len_v2")
dyn_batch_size = st.slider("Batch size", 4, 256, 64, step=4, key="dyn_batch_size_v2")

# Generate data
dyn_lengths = [random.randint(dyn_min_len, dyn_max_len) for _ in range(dyn_num_seq)]

# -------------------------
# Naive batching (unsorted)
# -------------------------
naive_batches = [
    dyn_lengths[i:i + dyn_batch_size]
    for i in range(0, dyn_num_seq, dyn_batch_size)
]

naive_total_pad = 0
naive_total_tokens = 0

for batch in naive_batches:
    bmax = max(batch)
    naive_total_pad += sum(bmax - l for l in batch)
    naive_total_tokens += bmax * len(batch)

# -------------------------
# Dynamic batching (sorted)
# -------------------------
dyn_lengths_sorted = sorted(dyn_lengths)

dyn_batches = [
    dyn_lengths_sorted[i:i + dyn_batch_size]
    for i in range(0, dyn_num_seq, dyn_batch_size)
]

dyn_total_pad = 0
dyn_total_tokens = 0

for batch in dyn_batches:
    bmax = max(batch)
    dyn_total_pad += sum(bmax - l for l in batch)
    dyn_total_tokens += bmax * len(batch)

# -------------------------
# Metrics
# -------------------------
real_tokens = sum(dyn_lengths)

naive_util = real_tokens / naive_total_tokens
dyn_util = real_tokens / dyn_total_tokens

pad_reduction = (naive_total_pad - dyn_total_pad) / naive_total_pad
util_gain = dyn_util - naive_util

# -------------------------
# Summary table
# -------------------------
st.subheader("Comparison")

comp_df = pd.DataFrame({
    "Metric": [
        "Total padded tokens",
        "Padding percentage",
        "Token utilization"
    ],
    "Naive batching": [
        str(naive_total_pad),
        f"{100 * naive_total_pad / naive_total_tokens:.2f}%",
        f"{100 * naive_util:.2f}%"
    ],
    "Dynamic batching": [
        str(dyn_total_pad),
        f"{100 * dyn_total_pad / dyn_total_tokens:.2f}%",
        f"{100 * dyn_util:.2f}%"
    ]
})

st.table(comp_df)

st.markdown(f"""
**Improvement**

- **Padding reduced by:** `{100 * pad_reduction:.2f}%`
- **Per Token Utilization increased by:** `{100 * util_gain:.2f}%`

Dynamic batching significantly cuts down wasted computation. Below, we visualize any two random batches. As you can see, each batch has a different size, but on a per-token level, we are seeing a lot more information passing through
""")

bin_size = 64

num_batches = len(dyn_batches)

low_start = int(0.2 * num_batches)
low_end   = int(0.4 * num_batches)

high_start = int(0.6 * num_batches)
high_end   = int(0.8 * num_batches)

idx_a = random.randint(low_start, max(low_start, low_end - 1))
idx_b = random.randint(high_start, max(high_start, high_end - 1))

batch_a = dyn_batches[idx_a]
batch_b = dyn_batches[idx_b]

def make_bar(seq_len, batch_max):
    total_bins = (batch_max // bin_size) + 1
    real_bins = int(seq_len // bin_size)
    pad_bins = total_bins - real_bins
    return "🟦" * real_bins + "⬜" * pad_bins

rows = []
max_rows = min(8, len(batch_a), len(batch_b))

max_a = max(batch_a)
max_b = max(batch_b)

for i in range(max_rows):
    rows.append({
        f"Batch {idx_a+1} (max={max_a})": make_bar(batch_a[i], max_a),
        f"Batch {idx_b+1} (max={max_b})": make_bar(batch_b[i], max_b),
    })

vis_df = pd.DataFrame(rows)

st.dataframe(vis_df, hide_index=True)

with st.expander("With transformers", expanded= False):
    st.markdown("""With HF transformers, you would likely use this""")
    st.code("""
    from transformers import Trainer

    trainer = Trainer(
        ...,
        group_by_length=True
    )
    """)

    st.markdown("For more fine grained control you might use the torch dataloader with specific collate function")

    st.code("""
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=...,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )
    """)

st.subheader("This isn't perfect")

st.markdown("""
Right now, we have managed to cut down the number of wasteful padding tokens, but we have a new problem. With varying length sizes, we have different GPU utilization per batch. For instance, if we have 4GB VRAM, and we are able to fit sequences of length 4096 comfortably, through dynamic batching the GPU may receive sequences of 100 tokens, or 4096 tokens. During shorter batches, we are not pushing the GPU to its limits, which means we need more cycles of training. At scale, with billions of tokens, this could easily result in several hundred extra GPU hours. 

There is another much more aggressive technique, **sequence packing**, which we will now discuss.
""")

st.divider()

st.header("Sequence Packing")

st.markdown("""
So far, we have dealt with two issues 

1. Varying lengths causing excess pad tokens - use dynamic batching to reduce pad tokens
2. Dynamic batching underutilizing peak GPU capacity 
            
We are already training the model to understand sequence boundaries with EOS tokens. Why not also combine multiple sequences into one single long sequence that consumes max GPU memory? This eliminates both the problem of computationally wasteful pad tokens, and uses GPU to maximal efficiency. However, there are still some fairly obvious issues to tackle 
            
1. When concatenating multiple sequences into one, what happens if the last one is too long and gets truncated to fit within context window? 
2. Since the sequences are separate, how are we letting the model know which tokens attend to which? Sequence 1 tokens must not interact with sequence 2, and vice versa. 
3. How are positional encodings maintained? Different sequences have different positions intra sequence. If we use one single constant positional encoding scheme, we are training the model for the wrong task 
            
As you can see, these are common problems we will come across. It is generally recommended to use production grade libraries to avoid any miniscule errors in coding out all of this, but at the very least we need to understand how each problem is being solved 

**Truncation**: Simple, whatever part of the sequence was truncated, add it to the beginning of the next input row in the batch. We do not waste any collected tokens this way. 

**Attention Masking**: We will need to write custom attention masks to enable specific attention flows. In particular for causal language modelling, we will see a lower triangular block diagonal matrix - sequence one will have a lower triangular attention pattern, and will not attend to any other seqeunce, same for other sequences, resulting in this shape 
            
**Positional Encoding**: Each sequence receives independent positional IDs. Our prior implementations did not cover this, we just applied encodings based on absolute positions. When truncated, the remaining sequence applied to the next input chunk will once again receive positional encodings based on original position within the sequence to preserve positional signals. Position is only reset to 0 when a new sequence boundary emerges. 
            
A great blog for understanding these would be https://huggingface.co/blog/sirluk/llm-sequence-packing
""")
st.header("Naive vs Dynamic vs Sequence Packing")

st.info("This is an oversimplified analysis only for demonstration")

st.markdown("""
We compare three strategies:

- **Naive batching**: random sequences → high variance → lots of padding  
- **Dynamic batching**: grouped by similar length → less padding  
- **Sequence packing**: keep appending sequences until context window is full → zero padding  

Each block = **64 tokens**  
🟦 = real tokens (naive/dynamic) | ⬜ = padding | 🟥🟧🟨🟩🟪🟫 = different packed sequences
""")

# -------------------
# Controls
# -------------------
cmp_num_samples = st.slider("Num Samples in Dataset", 128, 8192, 512, key="cmp_num_samples_final")
cmp_batch_size = st.slider("Batch size", 4, 32, 16, key="cmp_batch_size_final")
cmp_max_len = st.slider("Max sequence length", 128, 1024, 512, step=64, key="cmp_max_len_final")
cmp_min_len = st.slider("Min sequence length", 1, cmp_max_len // 4, 64, key="cmp_min_len_final")

# -------------------
# Generate pool
# -------------------
all_lengths = [random.randint(cmp_min_len, cmp_max_len) for _ in range(cmp_num_samples)]

# -------------------
# Naive: reshape into batches
# -------------------
trimmed = all_lengths[:len(all_lengths) - len(all_lengths) % cmp_batch_size]
naive_batches = np.reshape(trimmed, (-1, cmp_batch_size)).tolist()

# -------------------
# Dynamic: sort then reshape into batches
# -------------------
sorted_lengths = sorted(all_lengths)
sorted_trimmed = sorted_lengths[:len(sorted_lengths) - len(sorted_lengths) % cmp_batch_size]
dynamic_batches = np.reshape(sorted_trimmed, (-1, cmp_batch_size)).tolist()

# -------------------
# Sequence Packing
# -------------------
SEQ_COLORS = ["🟥", "🟧", "🟨", "🟩", "🟪", "🟫", "🔵", "🟢", "🔴", "🟤"]

packed_batches = []
cur_len = 0
cur_color = 0
colors = []
color_bin = []
cur_bin = []

for i in all_lengths:
    i = i + 1  # EOS token
    cur_len += i
    if cur_len <= cmp_max_len:
        color_bin.append(cur_color)
        cur_color += 1
        cur_bin.append(i)
    else:
        overflow_len = cur_len - cmp_max_len
        fit_len = i - overflow_len
        cur_bin.append(fit_len)
        color_bin.append(cur_color)
        packed_batches.append(cur_bin)
        colors.append(color_bin)
        cur_bin = [overflow_len]
        color_bin = [cur_color]
        cur_len = overflow_len
    cur_color += 1

if cur_bin:
    packed_batches.append(cur_bin)
    colors.append(color_bin)

# -------------------
# Batch selector
# -------------------
packed_grouped = [packed_batches[i:i + cmp_batch_size] for i in range(0, len(packed_batches), cmp_batch_size)]
colors_grouped = [colors[i:i + cmp_batch_size] for i in range(0, len(colors), cmp_batch_size)]

max_batch = max(len(naive_batches), len(dynamic_batches), len(packed_batches) // cmp_batch_size) - 1

batch_num = st.slider(
    "Batch number",
    0,
    max_batch,
    0,
    key="batch_num"
)

# -------------------
# Select batch for each strategy
# -------------------
naive_batch = naive_batches[batch_num] if batch_num < len(naive_batches) else None
dynamic_batch = dynamic_batches[batch_num] if batch_num < len(dynamic_batches) else None

packed_start = batch_num * cmp_batch_size
packed_end = packed_start + cmp_batch_size
packed_batch = packed_batches[packed_start:packed_end] if packed_start < len(packed_batches) else None
packed_color_batch = colors[packed_start:packed_end] if packed_start < len(colors) else None

naive_max = max(naive_batch) if naive_batch else 0
dyn_max = max(dynamic_batch) if dynamic_batch else 0

# -------------------
# Helpers
# -------------------
bin_size = 64

def make_bar(seq_len, max_len):
    total_bins = max(1, max_len // bin_size)
    real_bins = min(max(1, round(seq_len / bin_size)), total_bins)
    pad_bins = total_bins - real_bins
    return "🟦" * real_bins + "⬜" * pad_bins

def make_packed_bar(segments, seg_colors):
    bar = ""
    for idx, l in enumerate(segments):
        bins = max(1, round(l / bin_size))
        bar += SEQ_COLORS[seg_colors[idx] % len(SEQ_COLORS)] * bins
    return bar

# -------------------
# Visualization
# -------------------
st.subheader(f"Batch {batch_num}")

col1, col2, col3 = st.columns(3)
col1.markdown("**Naive**")
col2.markdown("**Dynamic**")
col3.markdown("**Packed**")

for i in range(cmp_batch_size):
    col1, col2, col3 = st.columns(3)
    with col1:
        if naive_batch and i < len(naive_batch):
            st.markdown(make_bar(naive_batch[i], naive_max))
        else:
            st.markdown("—")
    with col2:
        if dynamic_batch and i < len(dynamic_batch):
            st.markdown(make_bar(dynamic_batch[i], dyn_max))
        else:
            st.markdown("—")
    with col3:
        if packed_batch and i < len(packed_batch):
            st.markdown(make_packed_bar(packed_batch[i], packed_color_batch[i]))
        else:
            st.markdown("—")

# -------------------
# Full dataset stats
# -------------------

naive_pad_total = sum(
    (max(batch) * len(batch)) - sum(batch)
    for batch in naive_batches
)

dynamic_pad_total = sum(
    (max(batch) * len(batch)) - sum(batch)
    for batch in dynamic_batches
)

packed_pad_total = cmp_max_len - sum(packed_batches[-1]) if packed_batches else 0

naive_tokens_total = sum(max(batch) * len(batch) for batch in naive_batches)
dynamic_tokens_total = sum(max(batch) * len(batch) for batch in dynamic_batches)
packed_tokens_total = sum(sum(sum(bin) for bin in batch) for batch in packed_grouped)

naive_flops = sum((max(batch) * len(batch)) ** 2 for batch in naive_batches)
dynamic_flops = sum((max(batch) * len(batch)) ** 2 for batch in dynamic_batches)
packed_flops = sum((cmp_max_len * cmp_batch_size) ** 2 for batch in packed_grouped)

max_saturation = cmp_max_len * cmp_batch_size * 1.5

naive_saturation = np.mean([
    (max(batch) * len(batch)) / max_saturation * 100
    for batch in naive_batches
])

dynamic_saturation = np.mean([
    (max(batch) * len(batch)) / max_saturation * 100
    for batch in dynamic_batches
])

packed_saturation = np.mean([
    sum(sum(bin) for bin in batch) / max_saturation * 100
    for batch in packed_grouped
])

naive_pad_flops = sum((max(batch) * len(batch)) ** 2 - sum(batch) ** 2 for batch in naive_batches)
dynamic_pad_flops = sum((max(batch) * len(batch)) ** 2 - sum(batch) ** 2 for batch in dynamic_batches)
packed_pad_flops = sum((cmp_max_len * cmp_batch_size) ** 2 - (sum(sum(bin) for bin in batch)) ** 2 for batch in packed_grouped)

naive_compute_wasted = naive_pad_flops / naive_flops * 100
dynamic_compute_wasted = dynamic_pad_flops / dynamic_flops * 100
packed_compute_wasted = packed_pad_flops / packed_flops * 100

naive_gpu_hours = len(naive_batches)
dynamic_gpu_hours = len(dynamic_batches)
packed_gpu_hours = len(packed_grouped)

cost_df = pd.DataFrame({
    "Strategy": ["Naive", "Dynamic", "Packed"],
    "Total Batches": [len(naive_batches), len(dynamic_batches), len(packed_grouped)],
    "Total Tokens Processed": [naive_tokens_total, dynamic_tokens_total, packed_tokens_total],
    "Total Pad Tokens": [naive_pad_total, dynamic_pad_total, packed_pad_total],
    "Avg GPU Saturation": [
        f"{naive_saturation:.1f}%",
        f"{dynamic_saturation:.1f}%",
        f"{packed_saturation:.1f}%",
    ],
    "Compute Wasted on Padding": [
        f"{naive_compute_wasted:.1f}%",
        f"{dynamic_compute_wasted:.1f}%",
        f"{packed_compute_wasted:.1f}%",
    ],
    "Total FLOPs (relative)": [
        f"{naive_flops / naive_flops:.2f}x",
        f"{dynamic_flops / naive_flops:.2f}x",
        f"{packed_flops / naive_flops:.2f}x",
    ],
    "GPU Hours (relative)": [
        f"{naive_gpu_hours / naive_gpu_hours:.2f}x",
        f"{dynamic_gpu_hours / naive_gpu_hours:.2f}x",
        f"{packed_gpu_hours / naive_gpu_hours:.2f}x",
    ]
})

st.table(cost_df.set_index("Strategy"))

st.markdown(f"""
**Simplifying Assumptions For Above Analysis** 
- Sequence lengths are fully random, language datasets will follow other laws
- GPU is considered fully saturated at selected sequence length **{cmp_max_len} tokens × batch size {cmp_batch_size} = {cmp_max_len * cmp_batch_size} tokens per batch** times 1.5 for simplicity
- Padding is applied to **max sequence length within each batch**, not a global maximum
- FLOPs scale **quadratically with tokens processed per batch** — attention cost only, no feedforward or embedding costs. 
- GPU Hours are measured in **number of forward passes (batches)** — one batch = one unit of GPU time. This is obviously incorrect since different sequence lengths will lead to different processing times, but the principle holds.
- Compute wasted on padding = **compute for padded/total compute** per batch, averaged across all batches assuming quadratic cost for both
- FLOPs and GPU Hours are shown **relative to Naive** so Naive is always 1.00x
- The last packed bin may be partially filled — its remainder counts as padding
""")

st.header("Algorithms in practice") 

st.markdown("""
It's quite clear that of the options, sequence packing is the most effective. However, we must now concern ourselves with one other question - what order are the sequences packed in? 
            
In particular, this is important for training stability. We don't want to train the model on sequence lengths 1024 and 128k at the same time. We first want to provide smaller sequence lengths, stabilize it for that length, then progressively increase the size of sequence length. This way, the model can use and reshape its understanding of shorter sequence lengths on longer ones. It makes the training more stable, and converges faster. 
            
Briefly, we want algorithms that find the best fits for sequence lengths. This is a bin packing problem - given items of varying sizes, how do we sort and store them? 
            
1. **Naive Concatenation** ([`ConcatenativePacker`, NeMo-RL](https://docs.nvidia.com/nemo/rl/0.5.0/_modules/nemo_rl/data/packing/algorithms.html)): This is our baseline. We simply concat sequences until a bin is full, then start the next. No sorting, no consideration of fit.

2. **First Fit Decreasing (FFD)** ([`FirstFitDecreasingPacker`, NeMo-RL](https://docs.nvidia.com/nemo/rl/0.5.0/_modules/nemo_rl/data/packing/algorithms.html) / [NeMo SFT `first_fit_decreasing`](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/features/optimizations/sequence_packing.html)): Sort sequences longest-first, then assign each to the first bin it fits. O(n log n) to sort + O(n·m) for placement, where m is the number of bins.

3. **Best Fit Decreasing (BFD)** ([`strategy="ffd"` in TRL `pack_dataset`](https://github.com/huggingface/trl/blob/main/trl/data_utils.py) — note: [labelled FFD but actually implements BFD](https://github.com/huggingface/trl/issues/3645)): Sort longest-first, but place each sequence into the *tightest* bin — the one with least remaining capacity after placement. Slightly better packing efficiency than FFD.

4. **Modified First Fit Decreasing** ([`ModifiedFirstFitDecreasingPacker`, NeMo-RL](https://docs.nvidia.com/nemo/rl/0.5.0/_modules/nemo_rl/data/packing/algorithms.html) / [config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml)): The Johnson & Garey (1985) MFFD heuristic. Classifies sequences into large/medium/small/tiny and uses a 6-step packing strategy (classify → one bin per large item → add medium to large bins → add pairs of small items → greedily fit remaining → FFD on leftovers) for better bin utilisation than standard FFD. Adds hardware alignment constraints — sequences are padded to multiples of `cp_size * 2 * tp_size`. Recommended default for production pretraining in NeMo-RL.

5. **Length Bucketing (Dataset Decomposition)** ([`apple/ml-dataset-decomposition`](https://github.com/apple/ml-dataset-decomposition)): Rather than packing arbitrary sequences together, documents are sorted into power-of-2 length buckets. Each bucket contains sequences from a *single document*, so cross-document attention is eliminated by construction. A length-based curriculum can optionally control sampling probabilities across buckets — prioritising shorter buckets early in training — but by default sampling is uniform across all buckets.

Note that "short sequences first" in all of these methods means *filtering or bucketing by natural document length* — long documents are deferred to later in training, not truncated. Truncation when it appears is a side effect of bin-packing algorithms hitting a fixed context window, and is something all of the above methods try to minimise.
""")

st.markdown("""
After bin packing, the pipeline is:

1. **Concatenation** (offline): Token IDs within each bin are concatenated into a single 1D tensor of length `pack_size`, with metadata stored alongside to mark sub-sequence boundaries.
2. **Batching** (online, per step): The DataLoader stacks N bins into a microbatch and passes the tokens and boundary metadata to the model.
3. **Forward pass**: The attention mechanism uses the boundary metadata to ensure tokens only attend within their own sub-sequence. Loss is computed per-token and normalized per sub-sequence.

By the time a batch reaches your training loop, each sample is already a fully concatenated packed sequence.
""")

st.subheader("Implementation")

st.markdown("I will provide the complete code for the above when we come across the final training loop in the decoder model training section. That said, I would still recommend using standard implementations. They are bound to be more correct than anything I write out")