"""文本清洗与分词（复用自参考仓库 data_helper.py）。"""

import re


def clean_str_sst(string):
    """SST 数据集的 tokenization/清洗：保留字母数字与常见标点，统一小写。"""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """除 SST 外数据集的清洗：额外拆分 's、n't 等缩略与标点。"""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def line_to_words(line):
    """一行文本 → 词列表。"""
    clean_line = clean_str_sst(line.strip())
    return clean_line.split(" ")
