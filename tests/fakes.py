"""Test doubles: a toy vocab + fake tokenizers/backend with no torch model or network.

The toy vocab exercises every branch of the word-start / continuation / excluded
classification, and the fake backend emits scripted logits so the constrained beam
search produces deterministic words.
"""

import torch

from evals.backends.base import EvalBackend
from evals.beam_search import manual_top_k_words

# id -> surface text. Covers word-start (space+alpha), continuation (bare alpha),
# and excluded tokens (eos, space+digit, punctuation, bos).
TOY_VOCAB = {
    0: " cat",   # word-start
    1: " dog",   # word-start
    2: "s",      # continuation
    3: "ter",    # continuation
    4: "<eos>",  # excluded (not alpha) -- the EOS id
    5: " 123",   # excluded (space, rest not alpha)
    6: "!",      # excluded (punctuation)
    7: "x",      # continuation
    8: " the",   # word-start
    9: "z",      # continuation
    10: "<bos>", # excluded -- prompt sentinel
}
VOCAB_SIZE = len(TOY_VOCAB)
EOS_ID = 4
BOS_ID = 10

EXPECTED_WORD_START = {0, 1, 8}
EXPECTED_CONTINUATION = {2, 3, 7, 9}


class FakeTiktokenTokenizer:
    """Mimics a tiktoken tokenizer: exposes ``_mergeable_ranks`` (bytes -> id)."""

    def __init__(self):
        self._mergeable_ranks = {text.encode("utf-8"): tid for tid, text in TOY_VOCAB.items()}

    def encode(self, text, allowed_special=None):
        return [BOS_ID]

    def decode(self, token_ids):
        return "".join(TOY_VOCAB[i] for i in token_ids)


class FakeHFTokenizer:
    """Mimics a HuggingFace tokenizer: ``decode([id])`` + ``len`` + ``eos_token_id``."""

    eos_token_id = EOS_ID

    def __len__(self):
        return VOCAB_SIZE

    def encode(self, text, add_special_tokens=False):
        return [BOS_ID]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(TOY_VOCAB[i] for i in token_ids)


# Scripted next-token preferences keyed on the last token id. Unlisted last tokens
# fall back to preferring EOS so any beam can terminate.
_TRANSITIONS = {
    BOS_ID: {0: 5.0, 1: 4.0, 8: 3.0},  # start -> word-start tokens
    0: {2: 5.0, 4: 2.0},               # " cat" -> "s" (then "cats") or EOS ("cat")
    2: {4: 5.0},                       # "s" -> EOS
    1: {4: 5.0},                       # " dog" -> EOS
    8: {4: 5.0},                       # " the" -> EOS
}


class _FakeModel:
    def eval(self):
        return self


class FakeBackend(EvalBackend):
    """Backend with scripted logits; no real model or tokenizer needed.

    ``transitions`` maps a last-token id to ``{next_id: logit}`` preferences; unlisted
    last tokens fall back to preferring EOS. Pass a custom dict to script alternative
    generations (e.g. to test the word-start-constrained first token).
    """

    def __init__(self, device="cpu", transitions=None):
        super().__init__(model=_FakeModel(), tokenizer=None,
                         vocab_size=VOCAB_SIZE, eos_token_id=EOS_ID, device=device)
        self.transitions = transitions if transitions is not None else _TRANSITIONS

    def encode(self, text):
        return [BOS_ID]

    def decode(self, token_ids):
        return "".join(TOY_VOCAB[i] for i in token_ids)

    def _token_strings(self):
        for tid, text in TOY_VOCAB.items():
            yield text, tid

    def logits(self, beam_seqs):
        out = torch.full((beam_seqs.shape[0], self.vocab_size), -10.0, device=self.device)
        for b in range(beam_seqs.shape[0]):
            last = int(beam_seqs[b, -1].item())
            prefs = self.transitions.get(last, {EOS_ID: 5.0})
            for tid, val in prefs.items():
                out[b, tid] = val
        return out

    def top_k_words(self, prompt, continuation_ids, word_start_ids, k):
        return manual_top_k_words(self, prompt, continuation_ids, word_start_ids, k)
