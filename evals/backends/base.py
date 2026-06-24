"""The backend abstraction that isolates per-model mechanics.

A backend owns *model and tokenizer mechanics only*: loading, encode/decode kwargs,
how to read logits from a forward pass, vocab/EOS resolution, and how to enumerate
the vocabulary for constraint building, and how to produce the top-k single-word
completions (the canonical method, implemented per family). It deliberately does NOT
own the prompt style (a per-spec :class:`~evals.prompts.PromptStrategy`).
"""

from abc import ABC, abstractmethod

import torch


def classify_token_ids(items):
    """Partition vocabulary tokens into continuation and word-start id tensors.

    Args:
        items: iterable of ``(token_text, token_id)``.

    Returns:
        ``(continuation_ids, word_start_ids)`` as 1-D ``torch.long`` tensors (on CPU).
        A *word-start* token is space-prefixed then alphabetic ("` cat`"); a
        *continuation* token is bare alphabetic ("`cat`"). All other tokens (digits,
        punctuation, mixed) are excluded from both.
    """
    continuation_ids = []
    word_start_ids = []
    for token_text, token_id in items:
        if not token_text:
            continue
        starts_with_space = token_text.startswith(" ")
        rest_after_space = token_text[1:] if starts_with_space else None
        rest_is_alpha = rest_after_space.isalpha() if rest_after_space else False
        is_alpha = token_text.isalpha()

        if starts_with_space and rest_is_alpha:
            word_start_ids.append(token_id)
        elif not starts_with_space and is_alpha:
            continuation_ids.append(token_id)

    return (
        torch.tensor(continuation_ids, dtype=torch.long),
        torch.tensor(word_start_ids, dtype=torch.long),
    )


class EvalBackend(ABC):
    """Uniform interface over a loaded model + tokenizer for the eval harness."""

    def __init__(self, model, tokenizer, vocab_size, eos_token_id, device):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.device = device

    @abstractmethod
    def encode(self, text):
        """Encode ``text`` to a list of token ids."""

    @abstractmethod
    def decode(self, token_ids):
        """Decode a list of token ids back to a string."""

    @abstractmethod
    def _token_strings(self):
        """Yield ``(token_text, token_id)`` for every id in ``range(vocab_size)``."""

    @abstractmethod
    def logits(self, beam_seqs):
        """Return last-position logits ``[B, V]`` for a batch of token sequences."""

    @abstractmethod
    def top_k_words(self, prompt, continuation_ids, word_start_ids, k):
        """Return up to ``k`` distinct ``(word, score)`` single-word completions.

        Each family implements the canonical word-count-limiter method its own way
        (HF: ``generate`` + ``LogitsProcessor``; tiktoken: manual beam) but with
        functionally identical semantics. ``word_start_ids`` constrains the first
        token; ``continuation_ids`` are the only tokens that may extend the word
        thereafter (all other mass terminates it at EOS).
        """

    def build_constraint_ids(self):
        """Build (continuation_ids, word_start_ids) tensors on ``self.device``."""
        continuation_ids, word_start_ids = classify_token_ids(self._token_strings())
        return continuation_ids.to(self.device), word_start_ids.to(self.device)
