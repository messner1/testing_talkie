"""Guard the space-prefixing hazard across the tiktoken-bytes vs HF-decode paths.

VERIFIED hazard: in a byte-level BPE vocab (gpt2) the " cat" token's raw string is
"Ġcat" (no leading space). Classification MUST go through tokenizer.decode([id]) (which
normalises "Ġ" -> " "), as HFBackend does, not the raw token string. tiktoken instead
stores a literal-space byte (b" cat"), which the TiktokenBackend reads directly. Both
must converge on the same word-start vs continuation classification.
"""

import pytest

from evals.backends.base import classify_token_ids
from evals.backends.hf_backend import HFBackend
from evals.backends.tiktoken_backend import TiktokenBackend


def _word_start_and_continuation(backend):
    cont, ws = backend.build_constraint_ids()
    return set(ws.tolist()), set(cont.tolist())


def test_hf_byte_level_uses_decode_not_raw_token():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    space_cat = tok.encode(" cat", add_special_tokens=False)[0]
    bare_cat = tok.encode("cat", add_special_tokens=False)[0]

    # Regression guard: the RAW token string does NOT start with a space ("Ġcat"),
    # so a naive convert_ids_to_tokens classifier would MISCLASSIFY " cat".
    assert not tok.convert_ids_to_tokens(space_cat).startswith(" ")

    backend = HFBackend(model=None, tokenizer=tok,
                        vocab_size=len(tok), eos_token_id=tok.eos_token_id, device="cpu")
    ws, cont = _word_start_and_continuation(backend)

    # Via decode (what HFBackend uses) the classification is correct.
    assert space_cat in ws
    assert bare_cat in cont


def test_tiktoken_literal_space_classification():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        space_cat = enc.encode(" cat")[0]
        bare_cat = enc.encode("cat")[0]
        backend = TiktokenBackend(model=None, tokenizer=enc,
                                  vocab_size=enc.n_vocab, eos_token_id=0, device="cpu")
        ws, cont = _word_start_and_continuation(backend)
        assert space_cat in ws
        assert bare_cat in cont
    except ImportError:
        # Fall back to the literal-space-byte fake (faithful proxy for tiktoken bytes).
        from .fakes import EXPECTED_CONTINUATION, EXPECTED_WORD_START, FakeTiktokenTokenizer, VOCAB_SIZE
        backend = TiktokenBackend(model=None, tokenizer=FakeTiktokenTokenizer(),
                                  vocab_size=VOCAB_SIZE, eos_token_id=4, device="cpu")
        ws, cont = _word_start_and_continuation(backend)
        assert ws == EXPECTED_WORD_START and cont == EXPECTED_CONTINUATION


def test_cross_backend_agreement_on_toy_words():
    # Both representations of the same conceptual word-start/continuation distinction
    # classify identically through the shared classify_token_ids rule.
    hf_like = [(" cat", 0), ("cat", 1)]          # HF decode yields literal space
    tiktoken_like = [(" cat", 0), ("cat", 1)]    # tiktoken bytes decode to literal space
    cont_a, ws_a = classify_token_ids(hf_like)
    cont_b, ws_b = classify_token_ids(tiktoken_like)
    assert set(ws_a.tolist()) == set(ws_b.tolist()) == {0}
    assert set(cont_a.tolist()) == set(cont_b.tolist()) == {1}
