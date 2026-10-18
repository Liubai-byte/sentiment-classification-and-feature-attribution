"""Vocab building: assign a unique integer id to each word."""

from .cleaning import line_to_words

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(tsv_paths):
    """Build the vocab by scanning the TSV files.

    The first two positions are fixed as <pad>=0 and <unk>=1; the remaining words are numbered in order of first appearance.
    Returns word_to_idx (word -> int).
    """
    word_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    idx = len(word_to_idx)
    for path in tsv_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.rstrip("\n").rsplit("\t", 1)[0]
                for word in line_to_words(sentence):
                    if word not in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1
    return word_to_idx
