"""训练入口。

用法：uv run python train.py [--config configs/default.yaml] [--model bilstm]
"""

import argparse

from pytorch_sentiment_classification.training.trainer import train
from pytorch_sentiment_classification.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="训练 SST-2 情感分类器")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖配置里的 model（lstm/bilstm/cnn）")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    train(config)


if __name__ == "__main__":
    main()
