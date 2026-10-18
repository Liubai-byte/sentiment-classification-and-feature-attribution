"""Load pretrained word vectors (optional).

The assignment does not require pretrained vectors; when no embedding_path is provided, the model uses random initialization.
"""

import numpy as np


def load_bin_vec(fname, vocab):
    """Load the GoogleNews word2vec binary format (word + 300-dim float32)."""
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
    """Load the GloVe text format (each line is a word + dim floats)."""
    word_vecs = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            word = parts[0]
            if word in vocab and len(parts) == dim + 1:
                word_vecs[word] = np.asarray(parts[1:], dtype="float32")
    return word_vecs
