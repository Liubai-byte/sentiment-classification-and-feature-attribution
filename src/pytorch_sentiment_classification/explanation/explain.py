"""Integrated Gradients word-level attribution (Captum).

All configuration comes from configs/explain.yaml (model, checkpoint, sentences to explain, IG parameters, ablation).
Each run writes to outputs/explain/<run_name>/ without overwriting previous runs:
- ig_<i>.html: green/red highlighting + true/predicted labels + confidence;
- explain_results.md: summary of per-word signed scores;
- ablation.md: before/after comparison for replacing/deleting a high-contribution word.
"""

import argparse
import os
from datetime import datetime

import torch
from captum.attr import LayerIntegratedGradients

from ..models import get_model
from ..preprocess.cleaning import line_to_words
from ..preprocess.vocab import PAD_IDX, UNK_IDX, build_vocab
from ..utils.config import load_config

LABELS = ("negative", "positive")


def _reset_hidden_hook(module, inputs):
    """Reset the hidden state and sync batch_size before forward (IG calls forward repeatedly)."""
    module.batch_size = inputs[0].size(1)
    module.hidden = module.init_hidden()


def load_model(config, checkpoint, word_to_idx, batch_size=1):
    """Rebuild the model from config and load checkpoint weights."""
    use_gpu = torch.cuda.is_available()
    model = get_model(
        config["model"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        vocab_size=len(word_to_idx),
        label_size=2,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if use_gpu:
        model = model.cuda()
    return model


def _pad_ids_for_cnn(model, ids):
    """CNN kernels require seq_len >= max(kernel_size); pad short sequences on the left to that length.

    The PAD positions match the IG baseline (all 0 = PAD_IDX), so their attribution is always 0 and does not affect the result.
    Returns (padded ids, number of PADs prepended). Non-CNN models are returned unchanged with 0 PADs.
    """
    if not hasattr(model, "convs"):
        return ids, 0
    max_kernel = max(int(c.kernel_size[0]) for c in model.convs)
    pad_len = max_kernel - len(ids)
    if pad_len > 0:
        return [PAD_IDX] * pad_len + ids, pad_len
    return ids, 0


def predict_tokens(model, word_to_idx, tokens):
    """Predict on a token list, returning (predicted label, confidence)."""
    ids = [word_to_idx.get(w, UNK_IDX) for w in tokens]
    ids, _ = _pad_ids_for_cnn(model, ids)
    input_ids = torch.LongTensor(ids).unsqueeze(1)  # (seq_len, 1)
    model.eval()
    model.batch_size = 1
    model.hidden = model.init_hidden()
    with torch.no_grad():
        prob = torch.exp(model(input_ids))
        pred = int(prob.argmax(1).item())
        confidence = float(prob.max(1).values.item())
    return pred, confidence


def predict_sentence(model, word_to_idx, sentence):
    """Predict on a single sentence, returning (predicted label, confidence)."""
    return predict_tokens(model, word_to_idx, line_to_words(sentence))


def explain_sentence(model, word_to_idx, sentence, target=None, n_steps=50):
    """Run IG attribution on a single sentence, returning (tokens, scores, pred, confidence, target_idx)."""
    tokens = line_to_words(sentence)
    ids = [word_to_idx.get(w, UNK_IDX) for w in tokens]
    ids, pad_len = _pad_ids_for_cnn(model, ids)
    input_ids = torch.LongTensor(ids).unsqueeze(1)  # (seq_len, 1)

    handle = model.register_forward_pre_hook(_reset_hidden_hook)
    model.eval()

    with torch.no_grad():
        prob = torch.exp(model(input_ids))
        pred = int(prob.argmax(1).item())
        confidence = float(prob.max(1).values.item())

    target_idx = pred if target is None else target

    lig = LayerIntegratedGradients(model, model.embeddings)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        target=target_idx,
        n_steps=n_steps,
    )  # (seq_len, 1, embedding_dim)
    handle.remove()

    scores = attributions.sum(dim=2).squeeze(1).detach().cpu().numpy()  # (seq_len,)
    if pad_len:
        scores = scores[-len(tokens):]  # drop the prepended PADs and keep only the original sentence's words
    return tokens, scores, pred, confidence, target_idx


def render_html(sentence, tokens, scores, true_label, pred_label, confidence):
    """Positive contributions in green, negative in red, with intensity indicating relative strength; the header shows true/predicted labels and confidence."""
    max_abs = max(abs(float(s)) for s in scores) or 1.0
    spans = []
    for tok, s in zip(tokens, scores):
        s = float(s)
        intensity = min(abs(s) / max_abs, 1.0)
        if s >= 0:
            color = "rgba(0, 128, 0, %.2f)" % (0.15 + 0.85 * intensity)
        else:
            color = "rgba(220, 20, 20, %.2f)" % (0.15 + 0.85 * intensity)
        spans.append('<span style="background-color:%s">%s</span>' % (color, tok))

    true_name = LABELS[true_label] if true_label in (0, 1) else "?"
    pred_name = LABELS[pred_label] if pred_label in (0, 1) else "?"
    header = ("<p><b>True label:</b> %s &nbsp;|&nbsp; <b>Predicted label:</b> %s &nbsp;|&nbsp; "
              "<b>Confidence:</b> %.3f</p>" % (true_name, pred_name, confidence))
    body = "<p style='font-size:18px;line-height:2'>" + " ".join(spans) + "</p>"
    return ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
            + header + body + "</body></html>")


def load_true_labels(test_path):
    """Read test.tsv and return {sentence: true label}."""
    labels = {}
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sentence, label = line.rsplit("\t", 1)
            labels[sentence.strip()] = int(label)
    return labels


def run_ablation(model, word_to_idx, sentence, word, replacement):
    """Replace/delete one high-contribution word in a sentence and compare before/after predictions."""
    tokens = line_to_words(sentence)
    if word not in tokens:
        print("Warning: word '%s' is not in the sentence, skipping ablation" % word)
        return None

    pred_before, conf_before = predict_tokens(model, word_to_idx, tokens)

    new_tokens = list(tokens)
    idx = new_tokens.index(word)
    if replacement:
        new_tokens[idx] = replacement
    else:
        del new_tokens[idx]

    pred_after, conf_after = predict_tokens(model, word_to_idx, new_tokens)
    return {
        "word": word,
        "replacement": replacement if replacement else "(deleted)",
        "before_sentence": " ".join(tokens),
        "before_pred": pred_before,
        "before_conf": conf_before,
        "after_sentence": " ".join(new_tokens),
        "after_pred": pred_after,
        "after_conf": conf_after,
        "flipped": pred_before != pred_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients word-level attribution")
    parser.add_argument("--config", default="configs/explain.yaml", help="YAML config for explain")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Rebuild the model
    paths = [os.path.join(cfg["data_dir"], n) for n in ("train.tsv", "dev.tsv", "test.tsv")]
    word_to_idx = build_vocab(paths)
    model = load_model(cfg, cfg["checkpoint"], word_to_idx, batch_size=1)
    true_labels = load_true_labels(os.path.join(cfg["data_dir"], "test.tsv"))

    # Sentences: the sentences list + optional sentences_file
    sentences = [s for s in (cfg.get("sentences") or []) if s]
    if cfg.get("sentences_file"):
        with open(cfg["sentences_file"], "r", encoding="utf-8") as f:
            sentences += [ln.strip() for ln in f if ln.strip()]

    n_steps = cfg.get("n_steps", 50)
    target = cfg.get("target")

    # Output directory (each run does not overwrite previous runs)
    run_name = cfg.get("run_name") or (cfg["model"] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = os.path.join("outputs", "explain", run_name)
    os.makedirs(run_dir, exist_ok=True)

    report_lines = ["# IG Explanation Results", "", "> Run: %s" % run_name, ""]

    for i, sentence in enumerate(sentences):
        true_label = true_labels.get(sentence, None)
        tokens, scores, pred, confidence, target_idx = explain_sentence(
            model, word_to_idx, sentence, target=target, n_steps=n_steps)

        true_name = LABELS[true_label] if true_label in (0, 1) else "?"
        print("\nSentence: %s" % sentence)
        print("True label: %s / Predicted label: %s (confidence %.3f) / Explanation target: %s" % (
            true_name, LABELS[pred], confidence, LABELS[target_idx]))
        for tok, s in zip(tokens, scores):
            print("  %-16s %+.4f" % (tok, s))

        html_path = os.path.join(run_dir, "ig_%d.html" % i)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_html(sentence, tokens, scores, true_label, pred, confidence))
        print("Highlight visualization saved to: %s" % html_path)

        report_lines.append("## Example %d" % (i + 1))
        report_lines.append("")
        report_lines.append("- Sentence: %s" % sentence)
        report_lines.append("- True label: %s / Predicted label: %s / Confidence: %.3f" % (
            true_name, LABELS[pred], confidence))
        report_lines.append("- Word-level attribution (positive = supports the prediction, negative = against it):")
        report_lines.append("")
        report_lines.append("```")
        for tok, s in zip(tokens, scores):
            report_lines.append("  %-16s %+.4f" % (tok, float(s)))
        report_lines.append("```")
        report_lines.append("")

    report_path = os.path.join(run_dir, "explain_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\nExplanation summary saved to: %s" % report_path)

    # Ablation (before/after)
    ablation = cfg.get("ablation") or {}
    if ablation.get("enabled"):
        sentence = ablation.get("sentence") or (sentences[0] if sentences else "")
        word = ablation.get("word")
        replacement = ablation.get("replacement")
        if sentence and word:
            res = run_ablation(model, word_to_idx, sentence, word, replacement)
            if res:
                lines = [
                    "# Ablation Experiment (before/after)",
                    "",
                    "- Operated word: %s -> %s" % (res["word"], res["replacement"]),
                    "",
                    "| Case | Sentence | Predicted label | Confidence |",
                    "|------|----------|----------------|------------|",
                    "| Before | %s | %s | %.3f |" % (res["before_sentence"], LABELS[res["before_pred"]], res["before_conf"]),
                    "| After | %s | %s | %.3f |" % (res["after_sentence"], LABELS[res["after_pred"]], res["after_conf"]),
                    "",
                    "Prediction flipped: %s" % ("yes" if res["flipped"] else "no"),
                ]
                ab_path = os.path.join(run_dir, "ablation.md")
                with open(ab_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print("Ablation results saved to: %s" % ab_path)

    print("Output directory for this run: %s" % run_dir)


if __name__ == "__main__":
    main()
