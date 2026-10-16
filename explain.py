"""解释入口。

用法：uv run python explain.py --checkpoint outputs/checkpoints/best_model.pth --sentence "a warm , funny film"
"""

from pytorch_sentiment_classification.explanation.explain import main

if __name__ == "__main__":
    main()
