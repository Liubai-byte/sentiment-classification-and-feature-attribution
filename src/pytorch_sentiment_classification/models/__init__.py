"""Model definitions and factory.

Single entry point get_model(name, **kwargs); training/explanation scripts depend only on this name.
To add a new model, register it here without changing the callers.
"""

from .lstm import LSTMSentiment
from .bilstm import BiLSTMSentiment
from .cnn import CNNSentiment

_MODELS = {
    "lstm": LSTMSentiment,
    "bilstm": BiLSTMSentiment,
    "cnn": CNNSentiment,
}


def get_model(name, **kwargs):
    """Create a model by name. kwargs must include embedding_dim/hidden_dim/vocab_size/label_size/use_gpu/batch_size."""
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name} (available: {', '.join(_MODELS)})")
    return _MODELS[name](**kwargs)


__all__ = ["LSTMSentiment", "BiLSTMSentiment", "CNNSentiment", "get_model"]
