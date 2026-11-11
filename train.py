"""Training entry point for the ESC-50 project."""
import argparse

import torch

from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training setup ready on {device} with seed {args.seed}.")
    # TODO: Load ESC-50, build the model, and implement training.


if __name__ == "__main__":
    main()
