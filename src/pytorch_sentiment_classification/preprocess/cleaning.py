"""Text cleaning and tokenization (reused from the reference repo's data_helper.py)."""

import re


def clean_str_sst(string):
    """Tokenization/cleaning for the SST dataset: keep alphanumerics and common punctuation, lowercase everything."""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """Cleaning for datasets other than SST: additionally split contractions and punctuation such as 's and n't."""
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
    """One line of text -> word list."""
    clean_line = clean_str_sst(line.strip())
    return clean_line.split(" ")
