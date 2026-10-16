"""词表构建：给每个词分配唯一整数 id。"""

from .cleaning import line_to_words

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(tsv_paths):
    """遍历 TSV 建词表。

    前两个位置固定为 <pad>=0、<unk>=1，其余词按首次出现顺序编号。
    返回 word_to_idx（word -> int）。
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
