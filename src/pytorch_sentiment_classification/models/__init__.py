"""模型定义与工厂。

统一入口 get_model(name, **kwargs)，训练/解释脚本只依赖这个名字，
新增模型时只需在此注册，无需改动调用方。
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
    """按名字创建模型。kwargs 需包含 embedding_dim/hidden_dim/vocab_size/label_size/use_gpu/batch_size。"""
    if name not in _MODELS:
        raise ValueError(f"未知模型: {name}（可选：{', '.join(_MODELS)}）")
    return _MODELS[name](**kwargs)


__all__ = ["LSTMSentiment", "BiLSTMSentiment", "CNNSentiment", "get_model"]
