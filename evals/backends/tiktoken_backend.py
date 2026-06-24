"""Backend for Talkie models (tiktoken tokenizer, custom `talkie` loader)."""

from ..beam_search import manual_top_k_words
from .base import EvalBackend


class TiktokenBackend(EvalBackend):
    """Talkie base / web / it.

    The model's ``forward`` returns raw ``[B, V]`` logits. ``encode`` passes
    ``allowed_special="all"`` so the IT few-shot chat prompt's special tokens survive
    (harmless for base/web). Vocab size and EOS are supplied by the factory from the
    model spec (the IT EOS is resolved from the ``<|end|>`` token there).
    """

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def _token_strings(self):
        tokenizer = self.tokenizer
        if hasattr(tokenizer, '_mergeable_ranks'):
            items = tokenizer._mergeable_ranks.items()
        else:
            items = []
            for token_id in range(self.vocab_size):
                try:
                    items.append((tokenizer.decode_single_token_bytes(token_id), token_id))
                except Exception:
                    pass

        for token_bytes, token_id in items:
            try:
                token_text = token_bytes.decode('utf-8', errors='ignore')
            except Exception:
                continue
            yield token_text, token_id

    def logits(self, beam_seqs):
        return self.model.forward(beam_seqs)

    def top_k_words(self, prompt, continuation_ids, word_start_ids, k):
        return manual_top_k_words(self, prompt, continuation_ids, word_start_ids, k)
