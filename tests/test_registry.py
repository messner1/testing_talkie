"""Registry integrity and per-family field invariants."""

from evals.registry import MODEL_REGISTRY


def test_expected_models_present():
    assert set(MODEL_REGISTRY) == {"talkie-base", "talkie-web", "talkie-it", "typewriter"}


def test_family_specific_fields():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.name == name
        assert spec.family in ("talkie", "hf")
        assert hasattr(spec.prompt, "cloze_prompt")
        if spec.family == "talkie":
            assert spec.talkie_key and spec.tiktoken_style in ("base", "it")
            assert spec.hf_repo_id is None
        else:
            assert spec.hf_repo_id
            assert spec.talkie_key is None


def test_cutoffs_and_it_eos_resolution():
    assert MODEL_REGISTRY["typewriter"].cutoff_year == 1913
    assert MODEL_REGISTRY["talkie-base"].cutoff_year == 1930
    # IT defers EOS to load-time resolution of "<|end|>"; base/web pin 65535.
    assert MODEL_REGISTRY["talkie-it"].eos_token_id is None
    assert MODEL_REGISTRY["talkie-base"].eos_token_id == 65535
