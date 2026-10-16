"""训练与评估循环（复用自原 train_batch.py 的核心逻辑）。"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ..models import get_model
from ..preprocess.embeddings import load_bin_vec, load_glove
from ..preprocess.vocab import PAD_IDX
from ..utils.seed import set_seed
from .data import load_sst
from .metrics import classification_report_str, compute_metrics, get_accuracy


def train_epoch(model, train_iter, loss_function, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in tqdm(train_iter, desc="Train epoch " + str(epoch + 1)):
        sent, label = batch
        truth_res += label.tolist()
        model.batch_size = len(label)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_res += pred.data.max(1)[1].tolist()
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label = batch
        truth_res += label.tolist()
        model.batch_size = len(label)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_res += pred.data.max(1)[1].tolist()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ": loss %.2f acc %.1f" % (avg_loss, acc * 100))
    return acc


def predict(model, loader):
    """返回 (truth, preds) 两个列表，供最终指标计算。"""
    model.eval()
    truth, preds = [], []
    for batch in loader:
        sent, label = batch
        model.batch_size = len(label)
        model.hidden = model.init_hidden()
        pred = model(sent)
        truth += label.tolist()
        preds += pred.data.max(1)[1].tolist()
    return truth, preds


def build_embedding_matrix(word_to_idx, embedding_dim, embedding_path=None):
    """构建 embedding 权重矩阵：默认随机初始化，可选加载预训练向量覆盖。"""
    vocab_size = len(word_to_idx)
    pretrained = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim)).astype("float32")
    pretrained[PAD_IDX] = 0.0
    if embedding_path and os.path.exists(embedding_path):
        print("Load pretrained embeddings from", embedding_path)
        if embedding_path.endswith(".bin"):
            vectors = load_bin_vec(embedding_path, word_to_idx)
        else:
            vectors = load_glove(embedding_path, word_to_idx, embedding_dim)
        for word, vector in vectors.items():
            pretrained[word_to_idx[word]] = vector
    return pretrained


def train(config):
    set_seed(config.get("seed", 1))

    batch_size = config["batch_size"]
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    epochs = config["epochs"]
    label_size = 2  # 二分类：0=negative, 1=positive
    use_gpu = torch.cuda.is_available()

    train_loader, dev_loader, test_loader, word_to_idx = load_sst(config["data_dir"], batch_size)
    vocab_size = len(word_to_idx)

    model = get_model(
        config["model"],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        label_size=label_size,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )
    if use_gpu:
        model = model.cuda()

    pretrained = build_embedding_matrix(word_to_idx, embedding_dim, config.get("embedding_path"))
    model.embeddings.weight.data.copy_(torch.from_numpy(pretrained))

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.NLLLoss()

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "outputs", "checkpoints"))
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pth")

    print("Training...")
    best_dev_acc = 0.0
    for epoch in range(epochs):
        avg_loss, acc = train_epoch(model, train_loader, loss_function, optimizer, epoch)
        tqdm.write("Train: loss %.2f acc %.1f" % (avg_loss, acc * 100))
        dev_acc = evaluate(model, dev_loader, loss_function, "Dev")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_path)
            evaluate(model, test_loader, loss_function, "Test")

    print("Best dev acc: %.1f" % (best_dev_acc * 100))
    truth, preds = predict(model, test_loader)
    print(classification_report_str(truth, preds))
    print("Metrics:", compute_metrics(truth, preds))
    print("Checkpoint saved to", best_path)
    return best_dev_acc
