"""从测试集挑选用于解释的句子，记录真实/预测标签与置信度，供后续 explain 使用。

配置全部来自 configs/select.yaml（模型、checkpoint、挑选规则、输出目录名）。
用法：uv run python select_examples.py
"""

import argparse
import json
import os
import random
from datetime import datetime

from pytorch_sentiment_classification.explanation.explain import LABELS, load_model, predict_sentence
from pytorch_sentiment_classification.preprocess.vocab import build_vocab
from pytorch_sentiment_classification.utils.config import load_config
from pytorch_sentiment_classification.utils.seed import set_seed


def read_tsv(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sentence, label = line.rsplit("\t", 1)
            examples.append((sentence.strip(), int(label)))
    return examples


def main():
    parser = argparse.ArgumentParser(description="挑选测试句用于 IG 解释")
    parser.add_argument("--config", default="configs/select.yaml", help="select 的 YAML 配置")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    paths = [os.path.join(cfg["data_dir"], n) for n in ("train.tsv", "dev.tsv", "test.tsv")]
    word_to_idx = build_vocab(paths)
    model = load_model(cfg, cfg["checkpoint"], word_to_idx, batch_size=1)

    examples = read_tsv(os.path.join(cfg["data_dir"], "test.tsv"))

    correct, mistakes = [], []
    for sentence, true_label in examples:
        pred, confidence = predict_sentence(model, word_to_idx, sentence)
        item = {
            "sentence": sentence,
            "true_label": true_label,
            "pred_label": pred,
            "confidence": round(confidence, 4),
            "correct": pred == true_label,
        }
        (correct if pred == true_label else mistakes).append(item)

    random.shuffle(correct)
    random.shuffle(mistakes)
    selected = correct[: cfg["n_correct"]] + mistakes[: cfg["n_mistakes"]]
    if len(selected) < 5:
        rest = [x for x in correct + mistakes if x not in selected]
        random.shuffle(rest)
        selected += rest[: 5 - len(selected)]

    run_name = cfg.get("run_name") or (cfg["model"] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir = os.path.join("outputs", "select", run_name)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "selected_examples.json")
    txt_path = os.path.join(out_dir, "selected_sentences.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        for it in selected:
            f.write(it["sentence"] + "\n")

    n_correct = sum(1 for x in selected if x["correct"])
    n_mistakes = sum(1 for x in selected if not x["correct"])
    print("已选 %d 条（预测正确 %d / 预测错误 %d）" % (len(selected), n_correct, n_mistakes))
    print("| # | 句子 | 真实 | 预测 | 置信度 | 对/错 |")
    print("|---|------|------|------|--------|-------|")
    for i, it in enumerate(selected):
        print("| %d | %s | %s | %s | %.3f | %s |" % (
            i + 1, it["sentence"], LABELS[it["true_label"]], LABELS[it["pred_label"]],
            it["confidence"], "对" if it["correct"] else "错"))
    print("\n结构化结果: %s" % json_path)
    print("句子清单(供 explain sentences_file): %s" % txt_path)


if __name__ == "__main__":
    main()
