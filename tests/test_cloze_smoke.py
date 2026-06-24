"""CPU end-to-end smoke test of the real HuggingFace path.

Loads a tiny public model through HFBackend and runs run_cloze on a handful of
in-memory rows. Skips cleanly if transformers or the model download is unavailable.
This exercises the real HF logit extraction (`.logits[:, -1, :]`) and decode kwargs.
"""

import json

import pytest

pytest.importorskip("transformers")

from evals.backends.hf_backend import HFBackend  # noqa: E402
from evals.cloze import run_cloze  # noqa: E402
from evals.registry import ModelSpec  # noqa: E402
from evals.prompts import PlainPrefixPrompt  # noqa: E402

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def hf_backend():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:  # offline / download blocked
        pytest.skip(f"Could not load {TINY_MODEL}: {exc}")
    return HFBackend(model=model, tokenizer=tokenizer,
                     vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                     device="cpu")


def test_run_cloze_writes_expected_outputs(hf_backend, tmp_path):
    spec = ModelSpec(
        name="tiny", family="hf", cutoff_year=1930,
        prompt=PlainPrefixPrompt(), hf_repo_id=TINY_MODEL)
    dataset = [
        {"text": "The cat sat on the mat", "word": "mat",
         "sense_start_year": 1500, "entry_start_year": 1200},
        {"text": "She opened the door and stepped outside", "word": "outside",
         "sense_start_year": 1500, "entry_start_year": 1400},
        {"text": "A brand new gadget called the widget", "word": "widget",
         "sense_start_year": 1990, "entry_start_year": 1850},
    ]
    prefix = str(tmp_path / "cloze_tiny")

    df = run_cloze(hf_backend, spec, dataset, output_prefix=prefix, ks=[10, 20])

    assert len(df) == 3
    for col in ("text", "target_word", "year", "entry_start_year", "rank", "is_future",
                "correct@10", "correct@20"):
        assert col in df.columns

    summary = json.loads((tmp_path / "cloze_tiny_summary.json").read_text())
    assert summary["total_samples"] == 3
    assert "10" in summary["metrics_by_k"]   # fixed/ schema
    assert (tmp_path / "cloze_tiny_details.csv").exists()
