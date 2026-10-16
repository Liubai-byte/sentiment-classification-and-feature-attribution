"""轻量数据加载：读 TSV、建词表、token 化、padding、批处理（纯 PyTorch，不依赖 torchtext）。"""

import os

import torch
from torch.utils.data import DataLoader, Dataset

from ..preprocess.cleaning import line_to_words
from ..preprocess.vocab import PAD_IDX, UNK_IDX, build_vocab


class SST2Dataset(Dataset):
    """把单个 TSV 读到内存：每个样本是 (token id 列表, 标签 0/1)。"""

    def __init__(self, tsv_path, word_to_idx):
        self.examples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                sentence, label = line.rsplit("\t", 1)
                ids = [word_to_idx.get(w, UNK_IDX) for w in line_to_words(sentence)]
                self.examples.append((ids, int(label)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate_fn(batch):
    """把一批 (ids, label) 打包成 (padded_text (seq_len, batch), labels (batch,))。

    按长度降序排列，与模型取 ``lstm_out[-1]`` 的约定一致（参考仓库 BucketIterator 的做法）。
    """
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in batch]
    labels = torch.LongTensor([x[1] for x in batch])
    max_len = max(len(s) for s in sentences)
    padded = torch.full((max_len, len(batch)), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(sentences):
        padded[: len(s), i] = torch.LongTensor(s)
    return padded, labels


def load_sst(data_dir, batch_size):
    """返回 (train_loader, dev_loader, test_loader, word_to_idx)。"""
    train_path = os.path.join(data_dir, "train.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    word_to_idx = build_vocab([train_path, dev_path, test_path])

    train_ds = SST2Dataset(train_path, word_to_idx)
    dev_ds = SST2Dataset(dev_path, word_to_idx)
    test_ds = SST2Dataset(test_path, word_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # dev/test 各作为一整个 batch，与参考仓库一致
    dev_loader = DataLoader(dev_ds, batch_size=len(dev_ds), shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader, word_to_idx
