"""One-off conversion from raw SST-2 (stsa.binary.*) to TSV.

The repo already ships data/SST2/*.tsv, so you normally do not need to run this script;
only run it if you re-download the raw files from the Stanford website.
"""

import os

from .cleaning import line_to_words


def convert_to_tsv(train_name, dev_name, test_name, out_dir):
    """Convert stsa.binary.* text files to TSV, one ``sentence \\t label`` per line (label 0/1)."""
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
                # Raw format: the first token is the label, the rest is the sentence
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
