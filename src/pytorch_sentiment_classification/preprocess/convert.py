"""原始 SST-2（stsa.binary.*）→ TSV 的一次性转换。

仓库已随附 data/SST2/*.tsv，通常无需运行本脚本；
仅当你从 Stanford 官网重新下载原始文件后才需要。
"""

import os

from .cleaning import line_to_words


def convert_to_tsv(train_name, dev_name, test_name, out_dir):
    """把 stsa.binary.* 文本转成 TSV，每行形如 ``sentence \\t label``（标签 0/1）。"""
    os.makedirs(out_dir, exist_ok=True)
    for name in (train_name, dev_name, test_name):
        if not name:
            continue
        sentences = []
        with open(name, "r", encoding="utf-8") as f:
            for line in f:
                words = line_to_words(line)
                if len(words) < 2:
                    continue
                # 原始格式：第一个 token 是标签，其余是句子
                sentences.append(" ".join(words[1:]) + "\t" + words[0])
        basename = os.path.basename(name).split(".")[-1] + ".tsv"
        with open(os.path.join(out_dir, basename), "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
    return train_name, dev_name, test_name


if __name__ == "__main__":
    convert_to_tsv(
        "data/raw/stsa.binary.train",
        "data/raw/stsa.binary.dev",
        "data/raw/stsa.binary.test",
        "data/SST2",
    )
