"""Training entry point.

Usage: uv run python train.py [--config configs/default.yaml] [--model bilstm]
"""

import argparse

from pytorch_sentiment_classification.training.trainer import train
from pytorch_sentiment_classification.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train an SST-2 sentiment classifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file")
    parser.add_argument("--model", default=None, help="Override the model in the config (lstm/bilstm/cnn)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    train(config)


if __name__ == "__main__":
    main()
