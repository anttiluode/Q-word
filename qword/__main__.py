"""Command line receipts, with no optional dependencies beyond NumPy."""
import argparse
import json
from pathlib import Path
from .core import run_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-word: reproduce the predictive-state and oscillator-probe benchmark")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_receipt(steps=args.steps, seed=args.seed)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf8")
        print(f"Saved {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
