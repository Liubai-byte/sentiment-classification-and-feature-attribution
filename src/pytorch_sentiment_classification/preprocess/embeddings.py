"""加载预训练词向量（可选）。

作业不要求使用预训练向量；不提供 embedding_path 时模型用随机初始化。
"""

import numpy as np


def load_bin_vec(fname, vocab):
    """加载 GoogleNews word2vec 二进制格式（word + 300 维 float32）。"""
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode("latin-1")
                if ch == " ":
                    word = "".join(word)
                    break
                if ch != "\n":
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype="float32")
            else:
                f.read(binary_len)
    return word_vecs


def load_glove(fname, vocab, dim):
    """加载 GloVe 文本格式（每行 word + dim 个 float）。"""
    word_vecs = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            word = parts[0]
            if word in vocab and len(parts) == dim + 1:
                word_vecs[word] = np.asarray(parts[1:], dtype="float32")
    return word_vecs
