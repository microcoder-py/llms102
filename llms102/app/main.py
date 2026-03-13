# app/main.py

import streamlit as st

st.set_page_config(
    page_title="llms102",
    page_icon="🧠",
    layout="wide"
)

pg = st.navigation({
    "Home": [
        st.Page("pages/00_home.py", title="llms102", icon="🏠"),
    ],
    "Foundation": [
        st.Page("pages/foundation/01_language_modelling.py", title="Language Modelling", icon="📖"),
        st.Page("pages/foundation/02_tokenization.py", title="Tokenization", icon="🔤"),
        st.Page("pages/foundation/03_embeddings.py", title="Embeddings", icon="📐"),
    ],
    "Architecture": [
        st.Page("pages/architecture/04_attention.py", title="Attention", icon="👁️"),
        st.Page("pages/architecture/05_multi_head_attention.py", title="Multi-Head Attention", icon="👁️"),
        st.Page("pages/architecture/06_feedforward.py", title="Feedforward Layers", icon="⚡"),
        st.Page("pages/architecture/07_layer_norm_residuals.py", title="Layer Norm & Residuals", icon="⚖️"),
        st.Page("pages/architecture/08_positional_encoding.py", title="Positional Encoding", icon="📍"),
        st.Page("pages/architecture/09_transformer_block.py", title="Transformer Block", icon="🧱"),
    ],
    "Training": [
        st.Page("pages/training/10_loss_perplexity.py", title="Loss & Perplexity", icon="📉"),
        st.Page("pages/training/11_lr_schedules.py", title="Learning Rate Schedules", icon="📈"),
        st.Page("pages/training/12_gradient_flow.py", title="Gradient Flow", icon="🌊"),
        st.Page("pages/training/13_batch_packing.py", title="Batch & Sequence Packing", icon="📦"),
    ],
    "Causal": [
        st.Page("pages/causal/14_autoregressive_generation.py", title="Autoregressive Generation", icon="🔄"),
        st.Page("pages/causal/15_causal_masking_teacher_forcing.py", title="Causal Masking & Teacher Forcing", icon="🎭"),
        st.Page("pages/causal/16_decoder_only_models.py", title="Decoder-only Models", icon="🔓"),
    ],
    "Conditional": [
        st.Page("pages/conditional/17_conditional_generation.py", title="Conditional Generation", icon="🎯"),
        st.Page("pages/conditional/18_encoder_only_models.py", title="Encoder-only Models", icon="🔒"),
        st.Page("pages/conditional/19_classification.py", title="Classification", icon="🏷️"),
        st.Page("pages/conditional/20_sequence_to_sequence.py", title="Sequence to Sequence", icon="↔️"),
        st.Page("pages/conditional/21_sentence_embeddings.py", title="Sentence Embeddings", icon="🧬"),
    ],
    "Scaling": [
        st.Page("pages/scaling/22_kv_cache.py", title="KV Cache", icon="⚡"),
        st.Page("pages/scaling/23_flash_attention.py", title="Flash Attention", icon="🔦"),
        st.Page("pages/scaling/24_mixed_precision.py", title="Mixed Precision", icon="🎯"),
        st.Page("pages/scaling/25_qat.py", title="Quantization Aware Training", icon="🗜️"),
    ],
    "Post-training": [
        st.Page("pages/post_training/26_sft.py", title="Supervised Fine-tuning", icon="🎓"),
        st.Page("pages/post_training/27_lora.py", title="LoRA", icon="🔧"),
        st.Page("pages/post_training/28_distillation.py", title="Distillation", icon="🧪"),
        st.Page("pages/post_training/29_reward_modelling.py", title="Reward Modelling", icon="🏆"),
        st.Page("pages/post_training/30_orm_vs_prm.py", title="ORM vs PRM", icon="⚖️"),
        st.Page("pages/post_training/31_rlhf_ppo.py", title="RLHF & PPO", icon="🤝"),
        st.Page("pages/post_training/32_rlaif.py", title="RLAIF", icon="🤖"),
        st.Page("pages/post_training/33_grpo.py", title="GRPO", icon="📊"),
        st.Page("pages/post_training/34_dpo.py", title="DPO", icon="🎯"),
        st.Page("pages/post_training/35_rejection_sampling.py", title="Rejection Sampling", icon="🔍"),
        st.Page("pages/post_training/36_test_time_compute.py", title="Test Time Compute", icon="⏱️"),
    ],
    "Applied": [
        st.Page("pages/applied/37_inference_serving.py", title="Inference & Serving", icon="🚀"),
        st.Page("pages/applied/38_prompt_engineering.py", title="Prompt Engineering", icon="✍️"),
        st.Page("pages/applied/39_rag.py", title="RAG", icon="🔎"),
        st.Page("pages/applied/40_agents.py", title="Agents & Tool Use", icon="🤖"),
    ],
    "Appendix — Evaluations": [
        st.Page("pages/appendix/41_evaluation_fundamentals.py", title="Evaluation Fundamentals", icon="📋"),
        st.Page("pages/appendix/42_mmlu.py", title="MMLU", icon="📋"),
        st.Page("pages/appendix/43_humaneval.py", title="HumanEval", icon="💻"),
        st.Page("pages/appendix/44_hellaswag.py", title="HellaSwag", icon="🧠"),
        st.Page("pages/appendix/45_truthfulqa.py", title="TruthfulQA", icon="✅"),
        st.Page("pages/appendix/46_gsm8k.py", title="GSM8K", icon="🔢"),
        st.Page("pages/appendix/47_bigbench.py", title="BIG-Bench", icon="📏"),
        st.Page("pages/appendix/48_chatbot_arena.py", title="LMSYS Chatbot Arena", icon="🏟️"),
        st.Page("pages/appendix/49_llm_as_judge.py", title="LLM as Judge", icon="⚖️"),
        st.Page("pages/appendix/50_red_teaming.py", title="Red Teaming", icon="🔴"),
    ],
})

pg.run()