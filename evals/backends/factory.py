"""Build the right backend from a ModelSpec, encapsulating loading (difference (a)).

Heavy, environment-specific imports (`talkie`, `transformers`) are deferred into the
branch that needs them so importing this module never requires both ecosystems.
"""

from .hf_backend import HFBackend
from .tiktoken_backend import TiktokenBackend


def load_backend(spec, device, cache_dir="cache"):
    """Load ``spec``'s model + tokenizer and wrap them in an :class:`EvalBackend`."""
    if spec.family == "talkie":
        return _load_talkie(spec, device, cache_dir)
    if spec.family == "hf":
        return _load_hf(spec, device, cache_dir)
    raise ValueError(f"Unknown model family: {spec.family!r}")


def _load_talkie(spec, device, cache_dir):
    from talkie.config import MODELS
    from talkie.download import get_model_files
    from talkie.model import load_checkpoint
    from talkie.tokenizer import build_tokenizer

    model_spec = MODELS[spec.talkie_key]
    ckpt_path, vocab_path = get_model_files(spec.talkie_key, cache_dir=cache_dir)
    tokenizer = build_tokenizer(vocab_path, style=model_spec.style)

    model = load_checkpoint(
        str(ckpt_path), device, target_vocab_size=spec.target_vocab_size)

    eos_token_id = spec.eos_token_id
    if eos_token_id is None:  # IT: resolve the end-of-turn token from the tokenizer
        eos_token_id = tokenizer.encode("<|end|>", allowed_special="all")[0]

    return TiktokenBackend(
        model=model, tokenizer=tokenizer,
        vocab_size=spec.vocab_size, eos_token_id=eos_token_id, device=device)


def _load_hf(spec, device, cache_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo_id, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id)

    return HFBackend(
        model=model, tokenizer=tokenizer,
        vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id, device=device)
