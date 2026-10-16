"""Integrated Gradients 词级归因（Captum）。

对一条句子，计算每个词对「预测类别」的贡献大小，并输出带符号分数 +
一个绿色（正贡献）/红色（负贡献）高亮的 HTML。
"""

import argparse
import os

import torch
from captum.attr import LayerIntegratedGradients

from ..models import get_model
from ..preprocess.cleaning import line_to_words
from ..preprocess.vocab import UNK_IDX, build_vocab
from ..utils.config import load_config

LABELS = ("negative", "positive")


def _reset_hidden_hook(module, inputs):
    """forward 前重置 hidden 状态并同步 batch_size（IG 会反复调用 forward）。"""
    module.batch_size = inputs[0].size(1)
    module.hidden = module.init_hidden()


def explain_sentence(model, word_to_idx, sentence, target=None, n_steps=50):
    """对单句做 IG 归因，返回 (tokens, scores, pred, confidence, target_idx)。"""
    tokens = line_to_words(sentence)
    ids = [word_to_idx.get(w, UNK_IDX) for w in tokens]
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
    return tokens, scores, pred, confidence, target_idx


def render_html(tokens, scores):
    """正贡献绿色、负贡献红色，深浅表示相对强度。"""
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
    return ("<!doctype html><html><head><meta charset='utf-8'></head>"
            "<body><p style='font-size:18px;line-height:2'>" + " ".join(spans) + "</p></body></html>")


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients 词级归因")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="训练好的模型权重 .pth")
    parser.add_argument("--sentence", default=None, help="单条句子")
    parser.add_argument("--sentences", default=None, help="每行一条句子的文件")
    parser.add_argument("--target", type=int, default=None, help="解释的目标类别（默认用模型预测）")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = [os.path.join(config["data_dir"], n) for n in ("train.tsv", "dev.tsv", "test.tsv")]
    word_to_idx = build_vocab(paths)

    use_gpu = torch.cuda.is_available()
    model = get_model(
        config["model"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        vocab_size=len(word_to_idx),
        label_size=2,
        use_gpu=use_gpu,
        batch_size=1,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    if use_gpu:
        model = model.cuda()

    sentences = []
    if args.sentence:
        sentences.append(args.sentence)
    if args.sentences:
        with open(args.sentences, "r", encoding="utf-8") as f:
            sentences += [ln.strip() for ln in f if ln.strip()]

    out_dir = os.path.join("outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    for i, sentence in enumerate(sentences):
        tokens, scores, pred, confidence, target = explain_sentence(model, word_to_idx, sentence, args.target)
        print("\n句子: %s" % sentence)
        print("预测标签: %s（置信度 %.3f），解释目标类别: %s" % (LABELS[pred], confidence, LABELS[target]))
        for tok, s in zip(tokens, scores):
            print("  %-16s %+.4f" % (tok, s))

        out_path = os.path.join(out_dir, "ig_%d.html" % i)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_html(tokens, scores))
        print("高亮可视化已保存: %s" % out_path)
