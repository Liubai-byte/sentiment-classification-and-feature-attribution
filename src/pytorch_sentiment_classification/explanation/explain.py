"""Integrated Gradients 词级归因（Captum）。

配置全部来自 configs/explain.yaml（模型、checkpoint、要解释的句子、IG 参数、消融）。
每次运行输出到 outputs/explain/<run_name>/，互不覆盖：
- ig_<i>.html：绿/红高亮 + 真实/预测标签 + 置信度；
- explain_results.md：每个词的带符号分数汇总；
- ablation.md：高贡献词替换/删除的 before/after 对比。
"""

import argparse
import os
from datetime import datetime

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


def load_model(config, checkpoint, word_to_idx, batch_size=1):
    """按配置重建模型并加载 checkpoint 权重。"""
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


def predict_tokens(model, word_to_idx, tokens):
    """对 token 列表做预测，返回 (预测标签, 置信度)。"""
    ids = [word_to_idx.get(w, UNK_IDX) for w in tokens]
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
    """对单句做预测，返回 (预测标签, 置信度)。"""
    return predict_tokens(model, word_to_idx, line_to_words(sentence))


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


def render_html(sentence, tokens, scores, true_label, pred_label, confidence):
    """正贡献绿色、负贡献红色，深浅表示相对强度；头部显示真实/预测标签与置信度。"""
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
    header = ("<p><b>真实标签:</b> %s &nbsp;|&nbsp; <b>预测标签:</b> %s &nbsp;|&nbsp; "
              "<b>置信度:</b> %.3f</p>" % (true_name, pred_name, confidence))
    body = "<p style='font-size:18px;line-height:2'>" + " ".join(spans) + "</p>"
    return ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
            + header + body + "</body></html>")


def load_true_labels(test_path):
    """读 test.tsv，返回 {句子: 真实标签}。"""
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
    """替换/删除句子中的一个高贡献词，做 before/after 预测对比。"""
    tokens = line_to_words(sentence)
    if word not in tokens:
        print("警告：词 '%s' 不在句子中，跳过消融" % word)
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
        "replacement": replacement if replacement else "(删除)",
        "before_sentence": " ".join(tokens),
        "before_pred": pred_before,
        "before_conf": conf_before,
        "after_sentence": " ".join(new_tokens),
        "after_pred": pred_after,
        "after_conf": conf_after,
        "flipped": pred_before != pred_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients 词级归因")
    parser.add_argument("--config", default="configs/explain.yaml", help="explain 的 YAML 配置")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 重建模型
    paths = [os.path.join(cfg["data_dir"], n) for n in ("train.tsv", "dev.tsv", "test.tsv")]
    word_to_idx = build_vocab(paths)
    model = load_model(cfg, cfg["checkpoint"], word_to_idx, batch_size=1)
    true_labels = load_true_labels(os.path.join(cfg["data_dir"], "test.tsv"))

    # 句子：sentences 列表 + 可选 sentences_file
    sentences = [s for s in (cfg.get("sentences") or []) if s]
    if cfg.get("sentences_file"):
        with open(cfg["sentences_file"], "r", encoding="utf-8") as f:
            sentences += [ln.strip() for ln in f if ln.strip()]

    n_steps = cfg.get("n_steps", 50)
    target = cfg.get("target")

    # 输出目录（每次实验互不覆盖）
    run_name = cfg.get("run_name") or (cfg["model"] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = os.path.join("outputs", "explain", run_name)
    os.makedirs(run_dir, exist_ok=True)

    report_lines = ["# IG 解释结果", "", "> 运行: %s" % run_name, ""]

    for i, sentence in enumerate(sentences):
        true_label = true_labels.get(sentence, None)
        tokens, scores, pred, confidence, target_idx = explain_sentence(
            model, word_to_idx, sentence, target=target, n_steps=n_steps)

        true_name = LABELS[true_label] if true_label in (0, 1) else "?"
        print("\n句子: %s" % sentence)
        print("真实标签: %s / 预测标签: %s（置信度 %.3f）/ 解释目标类别: %s" % (
            true_name, LABELS[pred], confidence, LABELS[target_idx]))
        for tok, s in zip(tokens, scores):
            print("  %-16s %+.4f" % (tok, s))

        html_path = os.path.join(run_dir, "ig_%d.html" % i)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_html(sentence, tokens, scores, true_label, pred, confidence))
        print("高亮可视化已保存: %s" % html_path)

        report_lines.append("## 例子 %d" % (i + 1))
        report_lines.append("")
        report_lines.append("- 句子: %s" % sentence)
        report_lines.append("- 真实标签: %s / 预测标签: %s / 置信度: %.3f" % (
            true_name, LABELS[pred], confidence))
        report_lines.append("- 词级归因（正=支持预测，负=反对预测）:")
        report_lines.append("")
        report_lines.append("```")
        for tok, s in zip(tokens, scores):
            report_lines.append("  %-16s %+.4f" % (tok, float(s)))
        report_lines.append("```")
        report_lines.append("")

    report_path = os.path.join(run_dir, "explain_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n解释结果汇总已保存: %s" % report_path)

    # 消融（before/after）
    ablation = cfg.get("ablation") or {}
    if ablation.get("enabled"):
        sentence = ablation.get("sentence") or (sentences[0] if sentences else "")
        word = ablation.get("word")
        replacement = ablation.get("replacement")
        if sentence and word:
            res = run_ablation(model, word_to_idx, sentence, word, replacement)
            if res:
                lines = [
                    "# 消融实验（before/after）",
                    "",
                    "- 被操作词: %s → %s" % (res["word"], res["replacement"]),
                    "",
                    "| 情形 | 句子 | 预测标签 | 置信度 |",
                    "|------|------|----------|--------|",
                    "| 修改前 | %s | %s | %.3f |" % (res["before_sentence"], LABELS[res["before_pred"]], res["before_conf"]),
                    "| 修改后 | %s | %s | %.3f |" % (res["after_sentence"], LABELS[res["after_pred"]], res["after_conf"]),
                    "",
                    "预测是否翻转: %s" % ("是" if res["flipped"] else "否"),
                ]
                ab_path = os.path.join(run_dir, "ablation.md")
                with open(ab_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print("消融结果已保存: %s" % ab_path)

    print("本次实验输出目录: %s" % run_dir)


if __name__ == "__main__":
    main()
