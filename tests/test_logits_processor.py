"""The shared transform and its HF LogitsProcessor wrapper."""

import pytest
import torch

from evals.constraints import redistribute_to_eos


def test_redistribute_moves_non_continuation_mass_to_eos():
    # Vocab of 5; id 0 is EOS; ids 1,2 are continuation (kept); ids 3,4 are other
    # (word-start / punctuation -> redirected to EOS).
    continuation_ids = torch.tensor([1, 2])
    eos_id = 0
    logits = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0]])

    probs = torch.softmax(logits, dim=-1)
    out = redistribute_to_eos(logits, continuation_ids, eos_id)
    out_probs = torch.softmax(out, dim=-1)

    # Non-continuation ids (3, 4) are zeroed.
    assert out_probs[0, 3].item() == pytest.approx(0.0, abs=1e-9)
    assert out_probs[0, 4].item() == pytest.approx(0.0, abs=1e-9)
    # EOS absorbs exactly the non-continuation mass.
    moved = (probs[0, 3] + probs[0, 4]).item()
    assert out_probs[0, 0].item() == pytest.approx((probs[0, 0] + moved).item(), abs=1e-6)
    # Continuation tokens (1, 2) keep their original probability.
    assert out_probs[0, 1].item() == pytest.approx(probs[0, 1].item(), abs=1e-6)
    assert out_probs[0, 2].item() == pytest.approx(probs[0, 2].item(), abs=1e-6)
    # Probabilities still sum to 1.
    assert out_probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_score_formula_preserves_log_partition():
    # new_scores = log(new_probs) + logsumexp(logits); for a KEPT (continuation) token
    # this equals log(orig_prob) + logsumexp = orig_logit (up to the epsilon).
    continuation_ids = torch.tensor([3])
    logits = torch.tensor([[0.5, 3.0, -1.0, 2.0]])
    out = redistribute_to_eos(logits, continuation_ids, 0)
    # Token 3 is a continuation (untouched), so its score should match the original logit.
    assert out[0, 3].item() == pytest.approx(2.0, abs=1e-4)


def test_processor_restricts_first_token_then_redirects_non_continuation():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from evals.logits_processor import WordStartToEosLogitsProcessor

    continuation_ids = torch.tensor([3])      # only id 3 may extend the word
    word_start_ids = torch.tensor([1, 2])     # only ids 1,2 may begin the word
    proc = WordStartToEosLogitsProcessor(
        continuation_ids, word_start_ids, eos_token_id=0, prompt_length=3)
    scores = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0]])

    # At the prompt position (first generated token) only word-start ids survive; every
    # other token (incl. EOS=0) is masked to -inf so the answer must begin a fresh word.
    first = torch.zeros(1, 3)
    out_first = proc(first, scores)
    assert out_first[0, 1].item() == 1.0 and out_first[0, 2].item() == 1.0
    assert out_first[0, 0].item() == float("-inf")  # EOS masked: no empty first token
    assert out_first[0, 3].item() == float("-inf")
    assert out_first[0, 4].item() == float("-inf")

    # After the first generated token, the redistribute transform applies: only the
    # continuation token (3) keeps its mass; word-start tokens (1,2) and id 4 -> EOS.
    later = torch.zeros(1, 4)
    out = proc(later, scores)
    assert not torch.equal(out, scores)
    out_probs = torch.softmax(out, dim=-1)
    assert out_probs[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert out_probs[0, 2].item() == pytest.approx(0.0, abs=1e-9)
    assert out_probs[0, 3].item() == pytest.approx(torch.softmax(scores, dim=-1)[0, 3].item(), abs=1e-6)
