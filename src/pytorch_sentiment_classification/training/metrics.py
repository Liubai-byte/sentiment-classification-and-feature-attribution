"""Evaluation metrics: accuracy / precision / recall / F1."""

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    return sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)


def compute_metrics(truth, pred):
    """Return a dict of binary-classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(truth, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classification_report_str(truth, pred, target_names=("negative", "positive")):
    return classification_report(truth, pred, target_names=target_names, digits=4, zero_division=0)
