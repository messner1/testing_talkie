"""Manual constrained beam search — the tiktoken twin of the HF LogitsProcessor path.

tiktoken tokenizers can't use a HuggingFace ``LogitsProcessor``, so Talkie generates
single-word completions with this hand-rolled beam search. It is the faithful twin of
``WordStartToEosLogitsProcessor`` + ``model.generate(length_penalty=0)``: the first
generated token is constrained to word-start tokens
(:func:`~evals.constraints.restrict_to_word_start`), and every subsequent step applies
the shared :func:`~evals.constraints.redistribute_to_eos` transform (continue only on
bare-alphabetic continuation tokens; all other mass terminates the word at EOS).
"""

import torch

from .constraints import redistribute_to_eos, restrict_to_word_start


def filter_top_words(completed, decode, k):
    """Shared post-filter used by both backends.

    Sorts ``(token_list, score)`` completions by descending score, decodes each,
    strips/lowercases, keeps only pure-alphabetic words, deduplicates, and returns the
    top ``k`` as ``(word, score)``.
    """
    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    seen = set()
    results = []
    for tokens, score in completed:
        word = decode(tokens).strip().lower()
        if word and word.isalpha() and word not in seen:
            seen.add(word)
            results.append((word, score))
            if len(results) >= k:
                break
    return results


def beam_search_single(backend, input_ids, continuation_ids, word_start_ids,
                       num_beams=100, max_tokens=20):
    """Beam search for a single-word completion of ``input_ids``.

    Returns a list of ``(token_list, score)`` tuples, where ``token_list`` is the
    generated word's tokens (excluding the prompt) and ``score`` is its cumulative
    log-probability (no length penalty).
    """
    device = backend.device
    eos_token_id = backend.eos_token_id
    vocab_size = backend.vocab_size
    prompt_len = input_ids.shape[1]

    beam_seqs = input_ids
    beam_scores = torch.zeros(1, device=device)
    completed = []

    for step in range(max_tokens):
        if beam_seqs.shape[0] == 0:
            break

        with torch.no_grad():
            logits = backend.logits(beam_seqs)

        # First generated token is restricted to word-start tokens (the answer begins a
        # fresh word); later steps keep only bare-alphabetic continuation tokens and send
        # all other mass (word-start, punctuation, digits, ...) to EOS, terminating the
        # word at the next non-letter — the canonical method.
        if step == 0:
            logits = restrict_to_word_start(logits, word_start_ids)
        else:
            logits = redistribute_to_eos(logits, continuation_ids, eos_token_id)

        log_probs = torch.log_softmax(logits, dim=-1)
        candidate_scores = beam_scores.unsqueeze(-1) + log_probs
        flat_scores = candidate_scores.reshape(-1)

        valid_mask = torch.isfinite(flat_scores)
        if not valid_mask.any():
            break

        k = min(num_beams, int(valid_mask.sum().item()))
        topk_scores, topk_idx = flat_scores.topk(k)

        beam_idx = topk_idx // vocab_size
        token_idx = topk_idx % vocab_size

        eos_mask = (token_idx == eos_token_id)

        # Collect completions (EOS after at least one generated token).
        if eos_mask.any() and step > 0:
            for i in eos_mask.nonzero(as_tuple=True)[0]:
                src_beam = beam_idx[i].item()
                score = topk_scores[i].item()
                word_tokens = beam_seqs[src_beam, prompt_len:].tolist()
                if word_tokens:
                    completed.append((word_tokens, score))

        # Continue with non-EOS beams.
        cont_mask = ~eos_mask
        if not cont_mask.any():
            break

        cont_beam_idx = beam_idx[cont_mask]
        cont_token_idx = token_idx[cont_mask]
        cont_scores = topk_scores[cont_mask]

        beam_seqs = torch.cat([beam_seqs[cont_beam_idx], cont_token_idx.unsqueeze(-1)], dim=-1)
        beam_scores = cont_scores

        if len(completed) >= num_beams:
            break

    # Add any still-active beams as completions.
    if beam_seqs.shape[0] > 0 and len(completed) < num_beams:
        for i in range(min(beam_seqs.shape[0], num_beams - len(completed))):
            word_tokens = beam_seqs[i, prompt_len:].tolist()
            if word_tokens:
                completed.append((word_tokens, beam_scores[i].item()))

    return completed


def manual_top_k_words(backend, prompt, continuation_ids, word_start_ids, k=100):
    """Return up to ``k`` distinct ``(word, score)`` completions for ``prompt``."""
    token_ids = backend.encode(prompt)
    input_ids = torch.tensor([token_ids], device=backend.device)
    completed = beam_search_single(
        backend, input_ids, continuation_ids, word_start_ids, num_beams=k)
    return filter_top_words(completed, backend.decode, k)
