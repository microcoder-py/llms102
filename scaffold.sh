#!/bin/bash

# ─── root ────────────────────────────────────────────────────────────────────

mkdir -p llms102
cd llms102

# ─── library ─────────────────────────────────────────────────────────────────

mkdir -p src/llms102/core
mkdir -p src/llms102/architecture
mkdir -p src/llms102/training
mkdir -p src/llms102/causal
mkdir -p src/llms102/conditional
mkdir -p src/llms102/scaling
mkdir -p src/llms102/post_training
mkdir -p src/llms102/applied

touch src/llms102/__init__.py
touch src/llms102/core/__init__.py
touch src/llms102/core/language_model.py
touch src/llms102/core/tokenizer.py
touch src/llms102/core/embeddings.py

touch src/llms102/architecture/__init__.py
touch src/llms102/architecture/attention.py
touch src/llms102/architecture/multi_head_attention.py
touch src/llms102/architecture/feedforward.py
touch src/llms102/architecture/layer_norm.py
touch src/llms102/architecture/positional_encoding.py
touch src/llms102/architecture/transformer.py

touch src/llms102/training/__init__.py
touch src/llms102/training/loss.py
touch src/llms102/training/scheduler.py
touch src/llms102/training/trainer.py
touch src/llms102/training/packing.py

touch src/llms102/causal/__init__.py
touch src/llms102/causal/autoregressive.py
touch src/llms102/causal/causal_masking.py
touch src/llms102/causal/decoder.py

touch src/llms102/conditional/__init__.py
touch src/llms102/conditional/conditional.py
touch src/llms102/conditional/encoder.py
touch src/llms102/conditional/classification.py
touch src/llms102/conditional/seq2seq.py
touch src/llms102/conditional/sentence_embeddings.py

touch src/llms102/scaling/__init__.py
touch src/llms102/scaling/kv_cache.py
touch src/llms102/scaling/mixed_precision.py
touch src/llms102/scaling/qat.py

touch src/llms102/post_training/__init__.py
touch src/llms102/post_training/sft.py
touch src/llms102/post_training/lora.py
touch src/llms102/post_training/distillation.py
touch src/llms102/post_training/reward_model.py
touch src/llms102/post_training/orm.py
touch src/llms102/post_training/prm.py
touch src/llms102/post_training/rlhf_ppo.py
touch src/llms102/post_training/rlaif.py
touch src/llms102/post_training/grpo.py
touch src/llms102/post_training/dpo.py
touch src/llms102/post_training/rejection_sampling.py
touch src/llms102/post_training/test_time_compute.py

touch src/llms102/applied/__init__.py
touch src/llms102/applied/inference.py
touch src/llms102/applied/rag.py
touch src/llms102/applied/agents.py

# ─── app ─────────────────────────────────────────────────────────────────────

mkdir -p app/pages/foundation
mkdir -p app/pages/architecture
mkdir -p app/pages/training
mkdir -p app/pages/causal
mkdir -p app/pages/conditional
mkdir -p app/pages/scaling
mkdir -p app/pages/post_training
mkdir -p app/pages/applied
mkdir -p app/pages/appendix
mkdir -p app/components
mkdir -p app/utils

touch app/main.py
touch app/pages/00_home.py

touch app/pages/foundation/01_language_modelling.py
touch app/pages/foundation/02_tokenization.py
touch app/pages/foundation/03_embeddings.py

touch app/pages/architecture/04_attention.py
touch app/pages/architecture/05_multi_head_attention.py
touch app/pages/architecture/06_feedforward.py
touch app/pages/architecture/07_layer_norm_residuals.py
touch app/pages/architecture/08_positional_encoding.py
touch app/pages/architecture/09_transformer_block.py

touch app/pages/training/10_loss_perplexity.py
touch app/pages/training/11_lr_schedules.py
touch app/pages/training/12_gradient_flow.py
touch app/pages/training/13_batch_packing.py

touch app/pages/causal/14_autoregressive_generation.py
touch app/pages/causal/15_causal_masking_teacher_forcing.py
touch app/pages/causal/16_decoder_only_models.py

touch app/pages/conditional/17_conditional_generation.py
touch app/pages/conditional/18_encoder_only_models.py
touch app/pages/conditional/19_classification.py
touch app/pages/conditional/20_sequence_to_sequence.py
touch app/pages/conditional/21_sentence_embeddings.py

touch app/pages/scaling/22_kv_cache.py
touch app/pages/scaling/23_flash_attention.py
touch app/pages/scaling/24_mixed_precision.py
touch app/pages/scaling/25_qat.py

touch app/pages/post_training/26_sft.py
touch app/pages/post_training/27_lora.py
touch app/pages/post_training/28_distillation.py
touch app/pages/post_training/29_reward_modelling.py
touch app/pages/post_training/30_orm_vs_prm.py
touch app/pages/post_training/31_rlhf_ppo.py
touch app/pages/post_training/32_rlaif.py
touch app/pages/post_training/33_grpo.py
touch app/pages/post_training/34_dpo.py
touch app/pages/post_training/35_rejection_sampling.py
touch app/pages/post_training/36_test_time_compute.py

touch app/pages/applied/37_inference_serving.py
touch app/pages/applied/38_prompt_engineering.py
touch app/pages/applied/39_rag.py
touch app/pages/applied/40_agents.py

touch app/pages/appendix/41_evaluation_fundamentals.py
touch app/pages/appendix/42_mmlu.py
touch app/pages/appendix/43_humaneval.py
touch app/pages/appendix/44_hellaswag.py
touch app/pages/appendix/45_truthfulqa.py
touch app/pages/appendix/46_gsm8k.py
touch app/pages/appendix/47_bigbench.py
touch app/pages/appendix/48_chatbot_arena.py
touch app/pages/appendix/49_llm_as_judge.py
touch app/pages/appendix/50_red_teaming.py

touch app/components/__init__.py
touch app/components/attention_viz.py
touch app/components/embedding_viz.py
touch app/components/token_viz.py
touch app/components/loss_viz.py
touch app/components/gradient_viz.py
touch app/components/kv_cache_viz.py
touch app/components/reward_viz.py
touch app/components/quantization_viz.py
touch app/components/rag_viz.py

touch app/utils/__init__.py
touch app/utils/code_loader.py
touch app/utils/artifact_loader.py

# ─── artifacts ───────────────────────────────────────────────────────────────

mkdir -p artifacts/attention
mkdir -p artifacts/embeddings
mkdir -p artifacts/training
mkdir -p artifacts/scaling
mkdir -p artifacts/post_training

# ─── root files ──────────────────────────────────────────────────────────────

touch pyproject.toml
touch requirements.txt
touch README.md
touch .gitignore

# ─── placeholder content ─────────────────────────────────────────────────────

for f in $(find app/pages -name "*.py"); do
  echo "import streamlit as st
st.title('Coming soon')" > "$f"
done

# ─── git ─────────────────────────────────────────────────────────────────────

git init

echo "venv/
__pycache__/
*.pyc
.env
artifacts/
.DS_Store" > .gitignore

echo "streamlit
plotly
numpy
torch
transformers" > requirements.txt

echo "structure created successfully"