"""Unified evaluation harness for data-restricted language models.

Supports both Talkie (tiktoken tokenizer, custom `talkie` package loader) and
Typewriter / any HuggingFace causal LM behind a single backend abstraction, and
runs both the HPLM cloze leakage eval and the synthetic "-tron" composition eval.

Entry point: ``run_eval.py`` at the repo root.
"""
