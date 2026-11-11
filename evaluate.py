"""Evaluation entry point for the ESC-50 project."""
import argparse

from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"Evaluation setup ready with seed {args.seed}.")
    # TODO: Load a trained checkpoint and compute evaluation metrics.


if __name__ == "__main__":
    main()
