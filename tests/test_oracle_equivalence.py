"""Oracle equivalence: the manual tiktoken twin vs the real HF generate()+processor.

Both run against the SAME HuggingFace model, so any difference is purely the beam-search
*implementation*, not the tokenizer. If the manual twin matches the canonical HF path
here, the two production paths (Talkie manual / Typewriter generate) are functionally the
same method — which is what makes cross-family results comparable.
"""

import pytest

pytest.importorskip("transformers")

from evals.backends.hf_backend import HFBackend  # noqa: E402
from evals.beam_search import manual_top_k_words  # noqa: E402

TINY_MODEL = "sshleifer/tiny-gpt2"
PROMPTS = [
    "The cat sat on the",
    "She opened the door and stepped",
    "He picked up his pen and began to",
    "The capital of France is",
]
K = 8


@pytest.fixture(scope="module")
def backend():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:  # offline / download blocked
        pytest.skip(f"Could not load {TINY_MODEL}: {exc}")
    return HFBackend(model=model, tokenizer=tokenizer,
                     vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                     device="cpu")


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def test_manual_twin_matches_hf_generate(backend):
    continuation_ids, word_start_ids = backend.build_constraint_ids()

    top1_matches = 0
    jaccards = []
    for prompt in PROMPTS:
        native = [w for w, _ in backend.top_k_words(prompt, continuation_ids, word_start_ids, K)]
        manual = [w for w, _ in manual_top_k_words(
            backend, prompt, continuation_ids, word_start_ids, K)]

        assert native and manual, f"empty results for {prompt!r}"
        if native[0] == manual[0]:
            top1_matches += 1
        jaccards.append(_jaccard(native, manual))

    # The two implementations should agree on top-1 for most prompts and overlap
    # heavily on the top-k set.
    assert top1_matches >= len(PROMPTS) - 1
    assert sum(jaccards) / len(jaccards) >= 0.7
