# app/pages/00_home.py

import streamlit as st

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("llms102")
st.subheader("The last lesson you will need.")

st.markdown("")  # your mission — 3-4 sentences

col1, col2 = st.columns([1, 5])
col1.page_link("pages/foundation/01_language_modelling.py", label="Start here →")
col2.markdown("[GitHub →](https://github.com/you/llms102)")

st.divider()

# ─── About ───────────────────────────────────────────────────────────────────

st.header("About")

st.markdown("")  # why you built this
st.markdown("")  # philosophy — intuition before implementation
st.markdown("")  # how to use it — read the page, run the code yourself

st.divider()

# ─── Curriculum ──────────────────────────────────────────────────────────────

st.header("Curriculum")

st.markdown("")  # one sentence intro

def render_section(topics):
    for topic, (description, page) in topics.items():
        col1, col2, col3 = st.columns([2, 4, 1])
        col1.markdown(f"**{topic}**")
        col2.markdown(description)
        col3.page_link(page, label="Read →")

# ── Foundation ───────────────────────────────────────────────────────────────

st.subheader("Foundation")
render_section({
    "Language Modelling": ("The origin of language modelling, from Markov to neural networks.", "pages/foundation/01_language_modelling.py"),
    "Tokenization": ("How text becomes numbers, and why vocabulary design matters.", "pages/foundation/02_tokenization.py"),
    "Embeddings": ("How tokens become continuous vectors that capture meaning.", "pages/foundation/03_embeddings.py"),
})

st.divider()

# ── Architecture ─────────────────────────────────────────────────────────────

st.subheader("Architecture")
render_section({
    "Attention": ("The mechanism that lets every token relate to every other token.", "pages/architecture/04_attention.py"),
    "Multi-Head Attention": ("Running attention in parallel to capture multiple relationships.", "pages/architecture/05_multi_head_attention.py"),
    "Feedforward Layers": ("How the model transforms representations after attention.", "pages/architecture/06_feedforward.py"),
    "Layer Norm & Residuals": ("How deep networks stay stable and gradients keep flowing.", "pages/architecture/07_layer_norm_residuals.py"),
    "Positional Encoding": ("How the model knows where each token sits in a sequence.", "pages/architecture/08_positional_encoding.py"),
    "Transformer Block": ("Putting every component together into a complete block.", "pages/architecture/09_transformer_block.py"),
})

st.divider()

# ── Training ─────────────────────────────────────────────────────────────────

st.subheader("Training")
render_section({
    "Loss & Perplexity": ("What the model is optimising for and how we measure it.", "pages/training/10_loss_perplexity.py"),
    "Learning Rate Schedules": ("How we control the speed of learning over time.", "pages/training/11_lr_schedules.py"),
    "Gradient Flow": ("Why gradients vanish, explode, and how architecture prevents it.", "pages/training/12_gradient_flow.py"),
    "Batch & Sequence Packing": ("How we feed data efficiently to the model.", "pages/training/13_batch_packing.py"),
})

st.divider()

# ── Causal ───────────────────────────────────────────────────────────────────

st.subheader("Causal")
render_section({
    "Autoregressive Generation": ("How models generate text one token at a time.", "pages/causal/14_autoregressive_generation.py"),
    "Causal Masking & Teacher Forcing": ("How the model trains on sequences without seeing the future.", "pages/causal/15_causal_masking_teacher_forcing.py"),
    "Decoder-only Models": ("The architecture behind GPT and every major language model today.", "pages/causal/16_decoder_only_models.py"),
})

st.divider()

# ── Conditional ──────────────────────────────────────────────────────────────

st.subheader("Conditional")
render_section({
    "Conditional Generation": ("How models condition on context to produce targeted outputs.", "pages/conditional/17_conditional_generation.py"),
    "Encoder-only Models": ("How BERT-style models build rich bidirectional representations.", "pages/conditional/18_encoder_only_models.py"),
    "Classification": ("Using language models to assign labels to text.", "pages/conditional/19_classification.py"),
    "Sequence to Sequence": ("How encoder-decoder models translate, summarise, and transform.", "pages/conditional/20_sequence_to_sequence.py"),
    "Sentence Embeddings": ("How entire sequences become vectors for retrieval and search.", "pages/conditional/21_sentence_embeddings.py"),
})

st.divider()

# ── Scaling ──────────────────────────────────────────────────────────────────

st.subheader("Scaling")
render_section({
    "KV Cache": ("How inference stays fast as sequences grow longer.", "pages/scaling/22_kv_cache.py"),
    "Flash Attention": ("Why naive attention is memory inefficient and how to fix it.", "pages/scaling/23_flash_attention.py"),
    "Mixed Precision": ("How we train faster without sacrificing accuracy.", "pages/scaling/24_mixed_precision.py"),
    "Quantization Aware Training": ("How we compress models without destroying their capabilities.", "pages/scaling/25_qat.py"),
})

st.divider()

# ── Post-training ─────────────────────────────────────────────────────────────

st.subheader("Post-training")
render_section({
    "Supervised Fine-tuning": ("Teaching the model to follow instructions.", "pages/post_training/26_sft.py"),
    "LoRA": ("Fine-tuning large models efficiently with low-rank adapters.", "pages/post_training/27_lora.py"),
    "Distillation": ("Transferring knowledge from a large model to a smaller one.", "pages/post_training/28_distillation.py"),
    "Reward Modelling": ("Teaching the model what good outputs look like.", "pages/post_training/29_reward_modelling.py"),
    "ORM vs PRM": ("Rewarding outcomes versus rewarding reasoning steps.", "pages/post_training/30_orm_vs_prm.py"),
    "RLHF & PPO": ("Optimising the model against human preferences.", "pages/post_training/31_rlhf_ppo.py"),
    "RLAIF": ("Using AI feedback instead of human feedback.", "pages/post_training/32_rlaif.py"),
    "GRPO": ("Group relative policy optimisation and why it matters.", "pages/post_training/33_grpo.py"),
    "DPO": ("Direct preference optimisation as a simpler alternative to RLHF.", "pages/post_training/34_dpo.py"),
    "Rejection Sampling": ("Filtering model outputs to improve training data quality.", "pages/post_training/35_rejection_sampling.py"),
    "Test Time Compute": ("Making models smarter by thinking longer at inference.", "pages/post_training/36_test_time_compute.py"),
})

st.divider()

# ── Applied ───────────────────────────────────────────────────────────────────

st.subheader("Applied")
render_section({
    "Inference & Serving": ("How to deploy and serve language models efficiently.", "pages/applied/37_inference_serving.py"),
    "Prompt Engineering": ("How to communicate with language models effectively.", "pages/applied/38_prompt_engineering.py"),
    "RAG": ("Retrieval augmented generation for knowledge-intensive tasks.", "pages/applied/39_rag.py"),
    "Agents & Tool Use": ("How language models plan, act, and use external tools.", "pages/applied/40_agents.py"),
})

st.divider()

# ── Appendix ──────────────────────────────────────────────────────────────────

st.subheader("Appendix — Evaluations")
render_section({
    "Evaluation Fundamentals": ("What benchmarks measure and their limits.", "pages/appendix/41_evaluation_fundamentals.py"),
    "MMLU": ("Massive multitask language understanding.", "pages/appendix/42_mmlu.py"),
    "HumanEval": ("Code generation evaluation.", "pages/appendix/43_humaneval.py"),
    "HellaSwag": ("Commonsense reasoning.", "pages/appendix/44_hellaswag.py"),
    "TruthfulQA": ("Measuring hallucination.", "pages/appendix/45_truthfulqa.py"),
    "GSM8K": ("Grade school math reasoning.", "pages/appendix/46_gsm8k.py"),
    "BIG-Bench": ("Beyond the imitation game benchmark.", "pages/appendix/47_bigbench.py"),
    "LMSYS Chatbot Arena": ("Human preference evaluation at scale.", "pages/appendix/48_chatbot_arena.py"),
    "LLM as Judge": ("Using models to evaluate models.", "pages/appendix/49_llm_as_judge.py"),
    "Red Teaming": ("Finding failure modes systematically.", "pages/appendix/50_red_teaming.py"),
})

st.divider()

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
Built by [you](https://github.com/you) · 
[GitHub](https://github.com/you/llms102) · 
[HuggingFace](https://huggingface.co/spaces/you/llms102)
""")