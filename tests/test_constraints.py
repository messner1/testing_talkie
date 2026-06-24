"""Word-start / continuation classification across both backend paths."""

from evals.backends.base import classify_token_ids
from evals.backends.hf_backend import HFBackend
from evals.backends.tiktoken_backend import TiktokenBackend

from .fakes import (
    EXPECTED_CONTINUATION,
    EXPECTED_WORD_START,
    EOS_ID,
    VOCAB_SIZE,
    FakeHFTokenizer,
    FakeTiktokenTokenizer,
    TOY_VOCAB,
)


def test_classify_token_ids_partitions_toy_vocab():
    # classify_token_ids expects (token_text, token_id) pairs.
    cont, ws = classify_token_ids((text, tid) for tid, text in TOY_VOCAB.items())
    assert set(ws.tolist()) == EXPECTED_WORD_START
    assert set(cont.tolist()) == EXPECTED_CONTINUATION


def test_tiktoken_backend_constraint_ids():
    backend = TiktokenBackend(model=None, tokenizer=FakeTiktokenTokenizer(),
                              vocab_size=VOCAB_SIZE, eos_token_id=EOS_ID, device="cpu")
    cont, ws = backend.build_constraint_ids()
    assert set(ws.tolist()) == EXPECTED_WORD_START
    assert set(cont.tolist()) == EXPECTED_CONTINUATION


def test_hf_backend_constraint_ids():
    backend = HFBackend(model=None, tokenizer=FakeHFTokenizer(),
                        vocab_size=VOCAB_SIZE, eos_token_id=EOS_ID, device="cpu")
    cont, ws = backend.build_constraint_ids()
    assert set(ws.tolist()) == EXPECTED_WORD_START
    assert set(cont.tolist()) == EXPECTED_CONTINUATION
