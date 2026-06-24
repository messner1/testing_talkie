"""HuggingFace LogitsProcessor for the canonical single-word generation method.

Used with ``model.generate(..., logits_processor=[WordStartToEosLogitsProcessor(...)])``
for HF models (Typewriter). The first generated token is constrained to word-start
tokens (the answer begins a fresh word); every subsequent step keeps only bare-alphabetic
continuation tokens and redirects all other mass to EOS. Both branches call the shared
transforms in :mod:`evals.constraints`, making this the exact twin of the manual tiktoken
beam search.
"""

from transformers import LogitsProcessor

from .constraints import redistribute_to_eos, restrict_to_word_start


class WordStartToEosLogitsProcessor(LogitsProcessor):
    """Constrain the first generated token to word-start; continue only on continuation
    tokens (all other mass -> EOS) thereafter.

    Args:
        continuation_ids: 1-D tensor of bare-alphabetic continuation token ids (the only
            tokens allowed to extend the word after the first).
        word_start_ids: 1-D tensor of space-prefixed word-start token ids (the only
            tokens allowed to begin the answer word).
        eos_token_id: the EOS id that absorbs the redirected mass.
        prompt_length: number of prompt tokens; the first generated token (where
            ``input_ids.shape[1] == prompt_length``) is restricted to word-start tokens.
    """

    def __init__(self, continuation_ids, word_start_ids, eos_token_id, prompt_length):
        self.continuation_ids = continuation_ids
        self.word_start_ids = word_start_ids
        self.eos_token_id = eos_token_id
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        # First generated token (input still == prompt): must begin a fresh word.
        if input_ids.shape[1] <= self.prompt_length:
            return restrict_to_word_start(scores, self.word_start_ids.to(scores.device))
        return redistribute_to_eos(
            scores, self.continuation_ids.to(scores.device), self.eos_token_id)
