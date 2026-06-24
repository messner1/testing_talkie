#!/usr/bin/env python3
"""Unified entry point for the leakage + composition evals.

Replaces the per-model ``test_*.py`` scripts. Selects a model from the registry,
loads the appropriate backend, and runs the cloze and/or composition harness.

Examples:
    python run_eval.py --model talkie-base --task both
    python run_eval.py --model typewriter --task cloze --output-dir results/
    python run_eval.py --model talkie-it --task composition \
        --tron-cases tron_test_cases.jsonl
    # sanity-check inference over the first 500 cloze samples
    python run_eval.py --model typewriter --task cloze --limit 500
"""

import argparse
from pathlib import Path

from evals.backends import load_backend
from evals.registry import MODEL_REGISTRY


def resolve_device(choice):
    if choice != "auto":
        return choice
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run leakage/composition evals on a model")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY),
                        help="Model key from the registry")
    parser.add_argument("--task", default="both", choices=["cloze", "composition", "both"])
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--cloze-dataset", default="Hplm/historical-cloze")
    parser.add_argument("--tron-cases", default="tron_test_cases.jsonl")
    parser.add_argument("--cutoff", type=int, default=None,
                        help="Override the spec's default cutoff year (cloze only)")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sanity-check mode: evaluate only the first N cloze samples "
                             "(output is suffixed _subset{N} so it won't clobber a full run)")
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {spec.name} (family={spec.family}) | task={args.task} | device={device}")
    backend = load_backend(spec, device, cache_dir=args.cache_dir)
    print("Backend loaded.")

    if args.task in ("cloze", "both"):
        from datasets import load_dataset
        from evals.cloze import run_cloze

        dataset = load_dataset(args.cloze_dataset, split="test")
        suffix = ""
        if args.limit is not None:
            n = min(args.limit, len(dataset))
            dataset = dataset.select(range(n))
            suffix = f"_subset{n}"
            print(f"Sanity-check mode: first {n} cloze samples")
        run_cloze(
            backend, spec, dataset,
            output_prefix=str(output_dir / f"cloze_{spec.name}{suffix}"),
            ks=args.ks, cutoff_year=args.cutoff)

    if args.task in ("composition", "both"):
        from evals.composition import run_composition

        run_composition(
            backend, spec, args.tron_cases,
            output_prefix=str(output_dir / f"composition_{spec.name}"),
            ks=args.ks)


if __name__ == "__main__":
    main()
