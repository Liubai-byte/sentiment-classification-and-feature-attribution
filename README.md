# Sentiment Classification and Feature Attribution

Binary sentiment classification on the Stanford Sentiment Treebank (SST-2),
with word-level explanations via Integrated Gradients. Three interchangeable
encoders are available: LSTM, BiLSTM, and CNN. AIE1903 Explainable AI,
Homework 1.

## Quick start

Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync   # install dependencies and the package in editable mode
```

Every step is configured by a YAML file in `configs/` and run with:

```bash
uv run python train.py           
uv run python select_examples.py 
uv run python explain.py      
uv run python ablation.py    
```

- `train.py` trains the model (`--model lstm|bilstm|cnn`) and saves the
  best-dev checkpoint under `outputs/checkpoints/`.
- `select_examples.py` picks a mix of correct and mistaken test sentences.
- `explain.py` computes per-word Integrated Gradients and writes HTML
  highlights under `outputs/explain/`.
- `ablation.py` edits a highlighted word and shows the before/after prediction.

After training, copy the printed checkpoint path into the `checkpoint:` field of
`configs/explain.yaml`, `configs/select.yaml`, and `configs/ablation.yaml`.

## Project structure

```
configs/                          YAML config for each step
data/SST2/                        train.tsv, dev.tsv, test.tsv
src/pytorch_sentiment_classification/
  models/                         lstm.py, bilstm.py, cnn.py
  preprocess/                     cleaning, vocab, embeddings, convert
  training/                       data, trainer, metrics
  explanation/                    Integrated Gradients (explain.py)
  utils/                          config loading, random seed
train.py · select_examples.py · explain.py · ablation.py    CLI entry points
outputs/                          checkpoints, reports, select, explain, ablation
report/                           LaTeX report
```
