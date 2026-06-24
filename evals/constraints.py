"""The constrained-generation transforms shared by both model families.

These are the body of the canonical word-count-limiter method (Fittschen et al.),
adapted to the cloze task where the answer is known to be a single fresh word:

- :func:`restrict_to_word_start` is applied to the **first** generated token. The
  cloze prompt ends at a word boundary (it is ``rstrip()``-ed), so the answer's first
  token must be a space-prefixed word-start token; masking everything else spends the
  beam budget on legitimate fresh-word continuations instead of punctuation/digit/
  subword tokens that the ``isalpha()`` post-filter would discard anyway. This is
  lossless for the task (every valid single-word target begins with a word-start
  token) and is what keeps the finite-beam search an accurate top-k estimate.
- :func:`redistribute_to_eos` is applied to every **subsequent** token: only
  bare-alphabetic *continuation* tokens may extend the current word; the probability
  mass of everything else (word-initiating space-prefixed tokens, punctuation, digits,
  subword/special tokens) is redirected to EOS, so a beam terminates the word exactly
  when the model would emit anything that is not a letter continuing it.

The HuggingFace ``WordStartToEosLogitsProcessor`` (used with ``model.generate``) and
the manual tiktoken beam search both call these functions, so Talkie and Typewriter
are subject to functionally identical generation restrictions.
"""

import torch


def restrict_to_word_start(scores, word_start_ids):
    """Mask all non-word-start tokens to ``-inf`` (the first-token constraint).

    Args:
        scores: ``[B, V]`` logits for the first generated position.
        word_start_ids: 1-D tensor of space-prefixed word-start token ids (the only
            tokens allowed to *begin* the answer word).

    Returns:
        ``[B, V]`` logits with every non-word-start id set to ``-inf``; a following
        ``log_softmax`` then renormalizes over the word-start set. EOS is among the
        masked ids, so the first token can never be empty.
    """
    mask = torch.full_like(scores, float("-inf"))
    mask[:, word_start_ids] = scores[:, word_start_ids]
    return mask


def redistribute_to_eos(logits, continuation_ids, eos_token_id):
    """Move all non-continuation probability mass to EOS (the continuation constraint).

    Applied to every token after the first. The only tokens allowed to *extend* the
    current word are bare-alphabetic continuation tokens; the mass on everything else —
    word-start (space-prefixed) tokens, punctuation, digits, subword/special tokens — is
    redirected to EOS. The beam therefore terminates the word exactly when the model
    would emit anything that is not a letter continuing it.

    This is what makes the finite-beam search a faithful top-k single-word estimate:
    when the model's natural next token after, say, ``" cat"`` is a period or a fresh
    word, that mass flows to EOS and the candidate ``"cat"`` is *captured*, rather than
    grown into ``"cat."`` / ``"cat3"`` and later discarded by the ``isalpha()``
    post-filter (which would silently drop a word the model clearly favours). Genuine
    alpha continuations (``" cat"`` -> ``"egory"`` -> ``"category"``) are untouched, so
    longer words stay reachable.

    Args:
        logits: ``[B, V]`` raw logits for the current position.
        continuation_ids: 1-D tensor of bare-alphabetic continuation token ids (the only
            tokens that may extend the word; everything else is redirected to EOS).
        eos_token_id: the id that absorbs the redirected mass.

    Returns:
        ``[B, V]`` scores in log space: ``log(redistributed_probs) + logsumexp(logits)``,
        which preserves the original log-partition so cross-beam scores stay calibrated.
    """
    probs = torch.softmax(logits, dim=-1)
    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    allowed[continuation_ids] = True
    allowed[eos_token_id] = True

    disallowed_mass = probs[:, ~allowed].sum(dim=-1)
    new_probs = probs.clone()
    new_probs[:, ~allowed] = 0
    new_probs[:, eos_token_id] += disallowed_mass

    log_partition = torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.log(new_probs + 1e-10) + log_partition
