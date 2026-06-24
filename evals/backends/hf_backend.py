"""Backend for HuggingFace causal LMs (Typewriter and any AutoModelForCausalLM)."""

import torch

from ..beam_search import filter_top_words
from .base import EvalBackend


class HFBackend(EvalBackend):
    """Typewriter / any HuggingFace causal LM.

    The model is callable and returns an object exposing ``.logits``; we take the
    last position. ``encode`` suppresses added special tokens and ``decode`` skips
    them, so generated words decode cleanly. Vocab size and EOS are derived from the
    tokenizer (``len(tokenizer)`` / ``tokenizer.eos_token_id``).
    """

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _token_strings(self):
        for token_id in range(self.vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                continue
            yield token_text, token_id

    def logits(self, beam_seqs):
        return self.model(beam_seqs).logits[:, -1, :]

    def top_k_words(self, prompt, continuation_ids, word_start_ids, k):
        """Canonical method via HF beam search + WordStartToEosLogitsProcessor."""
        from ..logits_processor import WordStartToEosLogitsProcessor

        input_ids = torch.tensor([self.encode(prompt)], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        processor = WordStartToEosLogitsProcessor(
            continuation_ids, word_start_ids, self.eos_token_id, prompt_len)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=k,
                num_return_sequences=k,
                length_penalty=0.0,
                do_sample=False,
                max_new_tokens=20,
                min_new_tokens=1,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=[processor],
            )

        # Score each returned beam by its (length-penalty-free) sequence score and
        # reuse the shared decode/strip/isalpha/dedup filter.
        completed = [
            (seq[prompt_len:].tolist(), score.item())
            for seq, score in zip(out.sequences, out.sequences_scores)
        ]
        return filter_top_words(completed, self.decode, k)
