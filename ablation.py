"""Standalone ablation experiment: remove or replace a highly highlighted word
and observe whether the prediction changes.

Configuration comes from configs/ablation.yaml (checkpoint, sentence, target
word, mode, replacement). The result is written to
outputs/ablation/<run_name>/result.html, a self-contained page showing the
highlighted sentences plus the before/after predictions and confidence.

Usage:
    uv run python ablation.py
"""

import argparse
import os
from datetime import datetime

from pytorch_sentiment_classification.explanation.explain import (
    LABELS,
    explain_sentence,
    load_model,
    load_true_labels,
)
from pytorch_sentiment_classification.preprocess.cleaning import line_to_words
from pytorch_sentiment_classification.preprocess.vocab import build_vocab
from pytorch_sentiment_classification.utils.config import load_config


def label_name(idx):
    """Return the human-readable name of a label index, or '?' if unknown."""
    return LABELS[idx] if idx in (0, 1) else "?"


def color_span(token, score, max_abs):
    """Wrap a token in an HTML span colored by its (relative) attribution.

    Green = positive (supports the target class), red = negative (opposes it).
    Color depth encodes |score| relative to max_abs within the same sentence.
    """
    intensity = min(abs(score) / max_abs, 1.0)
    if score >= 0:
        color = "rgba(0, 128, 0, %.2f)" % (0.15 + 0.85 * intensity)
    else:
        color = "rgba(220, 20, 20, %.2f)" % (0.15 + 0.85 * intensity)
    return '<span style="background-color:%s">%s</span>' % (color, token)


def render_highlight(tokens, scores):
    """Render a token list as one line of highlighted HTML."""
    max_abs = max(abs(float(s)) for s in scores) or 1.0
    spans = [color_span(tok, float(s), max_abs) for tok, s in zip(tokens, scores)]
    return '<p style="font-size:18px;line-height:2">' + " ".join(spans) + "</p>"


def apply_edit(tokens, word, mode, replacement):
    """Return a copy of `tokens` with `word` replaced or deleted."""
    if word not in tokens:
        raise ValueError("word '%s' not found in the sentence" % word)

    edited = list(tokens)
    idx = edited.index(word)
    if mode == "replace":
        edited[idx] = replacement
    elif mode == "delete":
        del edited[idx]
    else:
        raise ValueError("mode must be 'replace' or 'delete', got '%s'" % mode)
    return edited


def main():
    parser = argparse.ArgumentParser(description="Ablation experiment (remove/replace a word)")
    parser.add_argument("--config", default="configs/ablation.yaml", help="ablation YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Rebuild the model from the checkpoint chosen in the YAML.
    paths = [os.path.join(cfg["data_dir"], n) for n in ("train.tsv", "dev.tsv", "test.tsv")]
    word_to_idx = build_vocab(paths)
    model = load_model(cfg, cfg["checkpoint"], word_to_idx, batch_size=1)
    true_labels = load_true_labels(os.path.join(cfg["data_dir"], "test.tsv"))

    sentence = cfg["sentence"]
    word = cfg["word"]
    mode = cfg["mode"]
    replacement = cfg.get("replacement")
    n_steps = cfg.get("n_steps", 50)
    target = cfg.get("target")

    true_label = true_labels.get(sentence)

    # Before: explain and predict on the original sentence.
    before_tokens, before_scores, before_pred, before_conf, _ = explain_sentence(
        model, word_to_idx, sentence, target=target, n_steps=n_steps)

    # After: apply the edit, then explain and predict on the edited sentence.
    edited_tokens = apply_edit(before_tokens, word, mode, replacement)
    after_sentence = " ".join(edited_tokens)
    after_tokens, after_scores, after_pred, after_conf, _ = explain_sentence(
        model, word_to_idx, after_sentence, target=target, n_steps=n_steps)

    flipped = before_pred != after_pred
    edit_desc = ("replaced '%s' with '%s'" % (word, replacement)
                 if mode == "replace" else "deleted '%s'" % word)

    # Unique output directory per run.
    run_name = cfg.get("run_name") or (cfg["model"] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir = os.path.join("outputs", "ablation", run_name)
    os.makedirs(out_dir, exist_ok=True)

    html = "\n".join([
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Ablation result</title></head><body>",
        "<h1>Ablation result</h1>",
        "<p><b>Edit:</b> %s</p>" % edit_desc,
        "<p><b>True label:</b> %s</p>" % label_name(true_label),
        "",
        "<h2>Before</h2>",
        render_highlight(before_tokens, before_scores),
        "<p><b>Prediction:</b> %s &nbsp;|&nbsp; <b>Confidence:</b> %.3f</p>" % (
            label_name(before_pred), before_conf),
        "",
        "<h2>After</h2>",
        render_highlight(after_tokens, after_scores),
        "<p><b>Prediction:</b> %s &nbsp;|&nbsp; <b>Confidence:</b> %.3f</p>" % (
            label_name(after_pred), after_conf),
        "",
        "<h2>Result</h2>",
        "<p><b>Prediction flipped:</b> %s</p>" % ("yes" if flipped else "no"),
        "</body></html>",
    ])

    html_path = os.path.join(out_dir, "result.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Edit: %s" % edit_desc)
    print("Before: %s (%.3f) -> After: %s (%.3f) | flipped: %s" % (
        label_name(before_pred), before_conf, label_name(after_pred), after_conf,
        "yes" if flipped else "no"))
    print("Result saved to: %s" % html_path)


if __name__ == "__main__":
    main()
