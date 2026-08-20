"""Model registry: friendly name -> ModelSpec.

Adding a model is adding one entry here. Each spec carries everything the harness
needs that is not derivable from the model object itself: family, cutoff year, prompt
strategy, and the family-specific loading parameters. (All models share the same
generation method — first token restricted to word-start tokens, then continue only on
bare-alphabetic continuation tokens with all other mass redirected to EOS — so there is
no per-model generation policy.)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .prompts import ITFewShotPrompt, PlainPrefixPrompt, PromptStrategy

# Talkie tiktoken vocab sizes (from the `talkie.tokenizer` package constants).
BASE_VOCAB_SIZE = 65536
IT_VOCAB_SIZE = BASE_VOCAB_SIZE + 4  # 65540

# Edgar is a local checkpoint, not a Hub repo, so its location is machine-specific.
# The default assumes edgar_synthetic sits beside this checkout; EDGAR_MODEL_PATH
# overrides it (the decode box keeps the weights elsewhere).
EDGAR_MODEL_PATH = os.environ.get(
    "EDGAR_MODEL_PATH",
    str(Path(__file__).resolve().parents[2]
        / "edgar_synthetic/edgar_student_teachers/edgar_clean/student"))


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # friendly CLI name / output label
    family: str                     # "talkie" | "hf"
    cutoff_year: int                # default knowledge cutoff for leakage split
    prompt: PromptStrategy

    # talkie-only
    talkie_key: Optional[str] = None        # key into `talkie.config.MODELS`
    tiktoken_style: Optional[str] = None    # "base" | "it"
    vocab_size: Optional[int] = None
    eos_token_id: Optional[int] = None      # None for IT => resolve "<|end|>" at load
    target_vocab_size: Optional[int] = None # passed to load_checkpoint (IT only)

    # hf-only
    hf_repo_id: Optional[str] = None


MODEL_REGISTRY = {
    "talkie-base": ModelSpec(
        name="talkie-base", family="talkie", cutoff_year=1930,
        prompt=PlainPrefixPrompt(),
        talkie_key="talkie-1930-13b-base", tiktoken_style="base",
        vocab_size=BASE_VOCAB_SIZE, eos_token_id=65535, target_vocab_size=None),

    "talkie-web": ModelSpec(
        name="talkie-web", family="talkie", cutoff_year=1930,
        prompt=PlainPrefixPrompt(),
        talkie_key="talkie-web-13b-base", tiktoken_style="base",
        vocab_size=BASE_VOCAB_SIZE, eos_token_id=65535, target_vocab_size=None),

    "talkie-it": ModelSpec(
        name="talkie-it", family="talkie", cutoff_year=1930,
        prompt=ITFewShotPrompt(),
        talkie_key="talkie-1930-13b-it", tiktoken_style="it",
        vocab_size=IT_VOCAB_SIZE, eos_token_id=None,  # resolve "<|end|>" at load
        target_vocab_size=IT_VOCAB_SIZE),

    "typewriter": ModelSpec(
        name="typewriter", family="hf", cutoff_year=1913,
        prompt=PlainPrefixPrompt(),
        hf_repo_id="typewriter-ai/typewriter-1913-7B-base"),

    # 631M Llama distilled on real 1786-1849 text plus synthetic augmentation whose
    # candidates were rejected on any out-of-vocabulary word, so the training word-type
    # inventory is the real corpus's. Byte-level BPE, so novel word forms remain
    # reachable. Separate replication, not a fourth column: no analysis script reads
    # this entry.
    "edgar": ModelSpec(
        name="edgar", family="hf", cutoff_year=1849,
        prompt=PlainPrefixPrompt(),
        hf_repo_id=EDGAR_MODEL_PATH),
}
