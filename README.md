# AIE1903 Explainable AI — Homework 1

## 情感分类与特征归因（Sentiment Classification and Feature Attribution）

本项目实现一个面向 **短电影评论句子的二分类情感分类器**，并用 **Integrated Gradients（积分梯度）** 对模型预测做词级别的可解释性分析，回答"模型到底靠哪些词做出判断"。

- **数据集**：SST-2（Stanford Sentiment Treebank 二分类版），正/负两类评论。
- **模型**：LSTM / BiLSTM / CNN 分类器（参考仓库为 LSTM + BiLSTM，CNN 可自行补充以做对比）。
- **可解释性**：使用 [Captum](https://captum.ai/) 的 Integrated Gradients，突出支持或反对预测情感的关键词。
- **参考起点**：[clairett/pytorch-sentiment-classification](https://github.com/clairett/pytorch-sentiment-classification)。

> 截止日期：2026-10-21。提交物：报告 + 代码 + 词级解释的可视化截图。

---

## 目录

1. [系统框架设计（软件工程视角）](#1-系统框架设计软件工程视角)
2. [目录结构](#2-目录结构)
3. [环境准备与工具选型](#3-环境准备与工具选型)
4. [分步实施流程](#4-分步实施流程)
5. [运行说明](#5-运行说明)
6. [结果记录模板](#6-结果记录模板)
7. [讨论要点](#7-讨论要点)
8. [常见问题与注意事项](#8-常见问题与注意事项)

---

## 1. 系统框架设计（软件工程视角）

### 1.1 设计目标

一个可解释性实验的本质是"**训练一个模型 → 打开它看内部依据 → 用证据支撑结论**"。为了让整个流程可复现、可扩展、可对比，系统在结构上遵循四条软件工程原则：

1. **单一职责（SRP）**：数据、模型、训练、解释四个环节各自独立成包，互不耦合。
2. **分层解耦（Layering）**：底层数据表示与上层训练/评估/解释逻辑分离，替换模型不影响解释层。
3. **配置与代码分离**：所有超参集中到 `configs/default.yaml`，训练逻辑与参数取值解耦，改参不改代码。
4. **可复现性（Reproducibility）**：固定随机种子、保存 checkpoint、记录实验版本，保证结果能重跑验证。

### 1.2 整体架构（数据流）

```
                        ┌─────────────────────────────────────────────┐
                        │            数据预处理层 (preprocess/)          │
                        │  cleaning / vocab / convert / embeddings     │
                        │  清洗 → 分词 → 建词表 → 词向量（可选预训练）    │
                        └──────────────────────┬──────────────────────┘
                                               │  sentence \t label
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │     加载层 (training/data.py，自写 DataLoader)│
                        │  SST2Dataset / collate_fn / DataLoader        │
                        │  索引化 + 按长度排序 + padding 成 batch        │
                        └──────────────────────┬──────────────────────┘
                                               │  (batch.text, batch.label)
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │            编码层 (Embedding)                 │
                        │  词向量：随机初始化，或预训练 word2vec/GloVe    │
                        └──────────────────────┬──────────────────────┘
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │          模型层 (models/)                     │
                        │  lstm / bilstm / cnn（经 get_model 工厂创建）  │
                        │  Embedding → 编码器 → Linear → softmax        │
                        └──────────────────────┬──────────────────────┘
                                               │  log-probabilities
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │   训练/评估层 (training/ + configs/default.yaml)│
                        │  前向 → NLLLoss → 反向 → Adam 更新 → 指标     │
                        └──────────────────────┬──────────────────────┘
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │          解释层 (explanation/explain.py)       │
                        │  Captum IntegratedGradients → 词级归因 → 可视化│
                        └─────────────────────────────────────────────┘
```

### 1.3 为什么这样分层

| 设计点 | 好处 | 具体体现 |
|--------|------|----------|
| 模型继承 `nn.Module` 且接口一致 | LSTM / BiLSTM / CNN 无缝切换，便于对比实验 | `forward(sentence) -> log_probs` 统一签名 |
| `models/__init__.py` 提供工厂 | 加新模型只改一处，训练脚本不 import 具体类 | `get_model(name, **kwargs)` 按字符串创建 |
| 训练脚本只负责编排 | 换优化器/损失函数不用改模型代码 | `training/trainer.py` 与模型解耦 |
| 超参走 YAML 配置 | 超参与逻辑分离，可记录、可对比多组实验 | `configs/default.yaml` + `utils/config.py` 加载 |
| 解释层独立 | IG 只需模型的 `forward` 与 embedding，不侵入训练代码 | `explanation/explain.py` 复用已训练模型 |
| checkpoint 按 dev 最优保存 | 避免过拟合，测试集只用于最终报告 | `best_model.pth` 按 `dev_acc` 触发保存 |

---

## 2. 目录结构

```
pytorch-sentiment-classification/
├── data/
│   └── SST2/
│       ├── train.tsv              # 训练集（已随仓库提供，无需下载）
│       ├── dev.tsv                # 验证集
│       └── test.tsv               # 测试集
│   # （可选）如需预训练词向量，放到 data/ 下，如 glove.6B.100d.txt
├── src/
│   └── pytorch_sentiment_classification/   # Python 包（src 布局，便于打包发布）
│       ├── __init__.py            # 包版本号
│       ├── preprocess/            # 数据预处理（对应原 data_helper.py 拆分）
│       │   ├── __init__.py
│       │   ├── cleaning.py        # clean_str_sst / line_to_words：清洗与分词
│       │   ├── vocab.py           # build_vocab：遍历语料建词表（<pad>=0、<unk>=1）
│       │   ├── convert.py         # （可选）从 Stanford 原始 stsa.binary.* 重建 TSV
│       │   └── embeddings.py      # 加载预训练词向量（word2vec/GloVe，可选）
│       ├── models/                # 模型定义
│       │   ├── __init__.py        # get_model(name, **kwargs) 工厂
│       │   ├── lstm.py            # LSTM 分类器
│       │   ├── bilstm.py          # 双向 LSTM 分类器
│       │   └── cnn.py             # （可选）CNN 分类器，用于对比
│       ├── training/              # 训练与评估
│       │   ├── __init__.py
│       │   ├── data.py            # SST2Dataset / collate_fn / load_sst（自写 DataLoader）
│       │   ├── trainer.py         # train_epoch / evaluate / train（原 train_batch.py）
│       │   └── metrics.py         # accuracy / precision / recall / F1
│       ├── explanation/           # 可解释性
│       │   ├── __init__.py
│       │   └── explain.py         # Captum IntegratedGradients + 词级高亮
│       └── utils/                 # 通用工具
│           ├── __init__.py
│           ├── config.py          # YAML 配置加载器（load_config）
│           └── seed.py            # 随机种子统一设置
├── configs/
│   └── default.yaml               # 超参集中配置
├── train.py                       # 训练入口（薄壳：解析参数 + 编排训练）
├── explain.py                     # 解释入口（薄壳）
├── outputs/
│   ├── checkpoints/               # best_model.pth（git 忽略）
│   └── figures/                   # 词级归因可视化 HTML
├── pyproject.toml                 # uv 项目配置（依赖声明）
├── uv.lock                        # uv 生成的锁定版本文件
├── .gitignore
└── README.md
```

> **关于 `__init__.py`**：它把目录标记成一个可 `import` 的 Python 包。有它才能写 `from pytorch_sentiment_classification.models import get_model`；它也可以是空文件，或承载包级公共代码（如 `models/__init__.py` 里的 `get_model` 工厂）、用 `__all__` 控制对外暴露。Python 3.3+ 没有它也能 import，但显式保留是标准做法，打包发布时（`uv build`）也必须有。
>
> **关于 src 布局**：包放在 `src/pytorch_sentiment_classification/` 下而非项目根目录，是 PyPA 推荐的 src 布局——它强迫你以「已安装」的方式使用自己的包（避免把项目根目录当成隐式 import 路径），发布到 GitHub/PyPI 时更干净。用 `uv sync` 会以 editable 模式安装本项目，因此根目录的 `train.py` 能直接 `from pytorch_sentiment_classification... import ...`。

### 2.1 旧文件 → 新位置映射

| 现在 | 改为 |
|------|------|
| `data_helper.py` | `src/pytorch_sentiment_classification/preprocess/{cleaning,vocab,convert,embeddings}.py` |
| `lstm.py` | `src/pytorch_sentiment_classification/models/lstm.py` |
| `bilstm.py` | `src/pytorch_sentiment_classification/models/bilstm.py` |
| （无） | `src/.../models/cnn.py` + `models/__init__.py`（工厂） |
| `train_batch.py` | `src/.../training/trainer.py` + 根目录 `train.py` |
| （无） | `src/.../training/data.py`（自写 DataLoader） |
| （无） | `src/.../explanation/explain.py` + 根目录 `explain.py` |
| （无） | `src/.../utils/{config,seed}.py` + `configs/default.yaml` |
| `data/SST2/*.tsv` | 保持不变（数据已就绪） |

---

## 3. 环境准备与工具选型

### 3.1 工具对比与选择

作业 PDF 推荐了 PyTorch 生态与 Captum；下面列出每个环节「参考仓库使用的工具」与「当前更常用的现代替代」，本项目优先采用现代、易维护的方案，同时保留参考仓库的经典做法。

| 环节 | PDF / 参考仓库使用 | 现代常用替代 | 本项目建议 |
|------|-------------------|-------------|-----------|
| 深度学习框架 | PyTorch | PyTorch 2.x（`torch.compile` 等） | **PyTorch** |
| 数据加载与分词 | `torchtext`（legacy：`data.Field`、`BucketIterator`） | Hugging Face `datasets` + `transformers` tokenizer | **自写轻量 `DataLoader`**（已实现，无 torchtext 依赖），进阶可用 HF |
| 预训练词向量 | （无要求，参考仓库用 word2vec） | GloVe、`transformers` 预训练 embedding | **可选**：随机初始化即可，需要时用 GloVe 小版 |
| 评估指标 | 手写 accuracy | `scikit-learn` `classification_report` | **sklearn**（acc/precision/recall/F1） |
| 训练进度 | `tqdm` | `tqdm` + Weights & Biases / TensorBoard | `tqdm` + （可选）**wandb** |
| 可解释性 | —（作业要求 IG） | **Captum** / SHAP / LIME | **Captum** `IntegratedGradients` |
| 可视化 | 自定义颜色高亮 | matplotlib / HTML 着色 | matplotlib + HTML 彩色高亮 |
| 配置管理 | 脚本顶部常量 / argparse | YAML / Hydra | **YAML**（`configs/default.yaml`） |
| 包与环境管理 | `pip` | **uv** / conda / poetry | **uv**（Rust 实现，快且现代） |
| 版本控制 | — | git | **git** |

### 3.2 用 uv 安装

```bash
# 1. 初始化项目（生成 pyproject.toml）
uv init

# 2. 添加依赖（写入 pyproject.toml 并安装；本项目已用自写 DataLoader，不再依赖 torchtext）
uv add torch tqdm numpy captum scikit-learn matplotlib pyyaml

# 3.（可选）实验跟踪
uv add wandb

# 4. 同步环境并安装本项目本身（src 布局的包以 editable 方式安装）
uv sync

# 之后所有命令都在项目环境里执行，无需手动 activate
uv run python train.py --model bilstm
```

> 说明：
> - 对 CPU / Mac（MPS），`uv add torch` 装的就是正确版本；只有要上 NVIDIA CUDA 才需额外指定 `--index-url`，本作业用不到。
> - 本项目用 **自写 DataLoader**（`training/data.py`）替代旧版 `torchtext` 的 `Field`/`BucketIterator`（新版 ≥0.13 已移除这些接口），因此 `pyproject.toml` 里没有 torchtext 依赖，也就避开了兼容问题。

---

## 4. 分步实施流程

下面按作业 PDF 的「Suggested steps」逐步展开，每一步都给出**做什么、用什么工具、怎么验证**。

### Step 0 — 数据就绪

数据已随仓库提供，**无需下载**：`data/SST2/` 下是 `train.tsv`（6919 行）、`dev.tsv`（871 行）、`test.tsv`（1820 行），每行形如：

```
a warm , funny , engaging film \t 1
```

> `preprocess/convert.py` 仅当你需要从 Stanford 官网的原始 `stsa.binary.*` 文件**重新构建** TSV 时才用，正常情况下直接跳过这一步。

### Step 1 — 数据准备（tokenize / 建词表 / padding）

**做什么**：把原始句子变成模型能吃的整数序列。

| 子步骤 | 实现位置 | 说明 |
|--------|----------|------|
| 清洗 + 分词 | `preprocess/cleaning.py` | 正则保留字母数字与标点，统一小写，按空格切词 |
| 建词表 | `preprocess/vocab.py` | `build_vocab` 遍历语料建 `word_to_idx`（`<pad>`=0、`<unk>`=1） |
| 索引化 + padding | `training/data.py` | 自写 `SST2Dataset` + `collate_fn` + `DataLoader`：batch 内按长度降序排序、pad 到最长 |

**工具**：`re`（正则）、PyTorch `Dataset`/`DataLoader`（自写，无 torchtext 依赖）。

> 设计要点：`collate_fn` 让 batch 内句子按长度排序后再 pad，使每个 batch 的 padding 更少，减少无效计算——这是文本任务里常用且重要的效率优化；同时与模型取 `lstm_out[-1]` 的约定一致。

### Step 2 — 模型构建（LSTM / BiLSTM / CNN）

**做什么**：定义分类网络。三个模型共享同一套骨架：`Embedding → 编码器 → 线性层 → softmax`，区别只在编码器。

- **LSTM**（`src/pytorch_sentiment_classification/models/lstm.py`）：`Embedding(300d) → LSTM(hidden=150) → Linear(150, 2) → log_softmax`。
- **BiLSTM**（`src/.../models/bilstm.py`）：LSTM 设为 `bidirectional=True`，输出维度翻倍为 `hidden*2`，能同时看左右上下文。
- **CNN**（可选，`src/.../models/cnn.py`）：`Embedding → Conv1d（多尺寸卷积核）→ 全局池化 → Linear`，用卷积核捕捉 n-gram 局部模式。

```python
# src/pytorch_sentiment_classification/models/cnn.py 骨架（用于对比实验，可选）
class CNNSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100, filter_sizes=(3,4,5)):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2)
    def forward(self, x):  # x: (seq_len, batch)
        x = self.embeddings(x).permute(1, 2, 0)   # (batch, dim, seq)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = self.dropout(torch.cat(x, dim=1))
        return F.log_softmax(self.fc(x), dim=1)
```

模型统一由工厂创建：

```python
# src/pytorch_sentiment_classification/models/__init__.py
_MODELS = {"lstm": LSTMSentiment, "bilstm": BiLSTMSentiment, "cnn": CNNSentiment}

def get_model(name, **kwargs):
    if name not in _MODELS:
        raise ValueError(f"unknown model: {name}")
    return _MODELS[name](**kwargs)
```

**工具**：PyTorch `nn.Module`（子类化，统一 `forward` 接口）。

### Step 3 — 训练与评估（指标）

**做什么**：初始化词向量（随机或预训练）、迭代训练、在 dev 上选最优模型、在 test 上报告结果。

关键超参集中在 `configs/default.yaml`：

```yaml
# configs/default.yaml
model: bilstm              # lstm / bilstm / cnn
embedding_dim: 300         # 词向量维度（用 GloVe 100d 时改成 100）
hidden_dim: 150            # LSTM 隐层维度
batch_size: 5
epochs: 20
learning_rate: 0.001
dropout: 0.5
seed: 1
data_dir: data/SST2
# embedding_path: data/glove.6B.100d.txt   # 可选：预训练词向量，不填则随机初始化
```

训练入口通过命令行覆盖默认模型，其余从 YAML 读取：

```bash
uv run python train.py --model bilstm    # 训练 BiLSTM
uv run python train.py --model lstm      # 训练 LSTM
```

**评估指标**：作业要求报告 test accuracy，并可选 precision / recall / F1。建议用 sklearn 统一计算（`training/metrics.py`）：

```python
from sklearn.metrics import classification_report
# 收集 truth_res / pred_res 后：
print(classification_report(truth_res, pred_res, target_names=['negative', 'positive'], digits=4))
```

> 设计要点：模型选择（checkpoint 保存）以 **dev 最优**为准，test 只在最终报告时用一次，避免在测试集上反复调参造成「测试集泄漏」。

### Step 4 — 记录测试句子的预测结果

**做什么**：挑若干测试句子，记录「真实标签 / 预测标签 / 置信度」，作为后续解释的素材。

置信度取 softmax 后两类概率中的最大值：

```python
model.eval()
probs = torch.exp(model(sentence))       # log_softmax → softmax
confidence, pred = probs.max(1)
```

> 记录建议：正误样本都挑（既要有预测对的，也要有预测错的），这样第 5 步的解释才更有信息量。

### Step 5 — Integrated Gradients 词级归因（核心）

**做什么**：用 Captum 计算每个输入词对预测结果的贡献大小，并可视化。

**原理（一句话）**：Integrated Gradients 从「全零基线」到「真实输入」之间沿着直线路径对梯度做积分，把预测值公平地"分摊"到每个输入 token 上：

```
IG_i(x) = (x_i − x'_i) · ∫₀¹  ∂F(x' + α(x − x')) / ∂x_i  dα
```

**工具**：Captum 的 `IntegratedGradients`（或 `LayerIntegratedGradients` 直接对 embedding 层求归因，更稳定）。

```python
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz

ig = LayerIntegratedGradients(model, model.embeddings)
# input_ids: (seq_len, batch) 的 token id；baselines 为同形状的全 0
attributions = ig.attribute(inputs=input_ids, baselines=torch.zeros_like(input_ids),
                            target=pred_label, n_steps=50)
# 求和/取范数得到每个词的标量得分，再按符号正负染色
```

**可视化**：两种常见做法——
- **HTML 着色**：正贡献（支持预测）标绿、负贡献（反对预测）标红，深浅代表强度（适合报告截图）。
- **matplotlib 条形图**：每个词一个带符号的分数，直观展示正负贡献。

> 设计要点：`LayerIntegratedGradients` 直接作用在 embedding 层，得到的归因是词向量级别的，求和后即为每个词的贡献，比在输入 id 上求更平滑、更易解释。

### Step 6 — 消融验证（替换/删除高贡献词）

**做什么**：挑一个高贡献词，把它**删除**或**替换**成中性词，观察预测是否翻转，做前后对比。

例如：句子 "a warm, funny, engaging film" 中 `engaging` 被高亮，把 `engaging` 替换为 `the` 或删除后，看置信度/标签变化。

| 情形 | 句子 | 预测标签 | 置信度 |
|------|------|----------|--------|
| 修改前 | a warm, funny, **engaging** film | positive | 0.87 |
| 修改后 | a warm, funny, the film | negative | 0.61 |

> 这验证了归因结果确实对应模型行为，而不只是"看起来合理"。

### Step 7 — 撰写报告与提交

按 PDF 要求整理：预处理说明 → 模型与测试结果 → 至少 5 个解释样例 → 简短结论；附上运行说明与可视化截图。

---

## 5. 运行说明

```bash
# 1. 安装依赖并安装本项目本身（首次；editable 安装使 src/ 下的包可 import）
uv sync

# 2. 训练（模型与超参由 configs/default.yaml 控制，命令行可覆盖 --model）
uv run python train.py --model lstm
uv run python train.py --model bilstm
uv run python train.py --model cnn        # 可选对比

# 3. 解释：对若干句子跑 Integrated Gradients 并可视化
uv run python explain.py --checkpoint outputs/checkpoints/best_model.pth --sentences examples.txt
```

> 数据已在 `data/SST2/`，无需预处理或下载。训练过程中会打印 `Train / Dev / Test` 的 loss 与 accuracy，最优模型按 dev 精度保存到 `outputs/checkpoints/best_model.pth`，可视化图保存到 `outputs/figures/`。

---

## 6. 结果记录模板

### 6.1 模型性能

| 模型 | Test Accuracy | Precision | Recall | F1 |
|------|---------------|-----------|--------|----|
| LSTM | — | — | — | — |
| BiLSTM | — | — | — | — |
| CNN（可选） | — | — | — | — |

### 6.2 解释样例（至少 5 个，含正误例）

| # | 句子 | 真实标签 | 预测标签 | 置信度 | 高亮词 | 是否符合直觉 |
|---|------|----------|----------|--------|--------|--------------|
| 1 | … | pos/neg | pos/neg | 0.xx | … | 是/否 |

---

## 7. 讨论要点

作业要求结合**自己模型的真实例子**讨论，不要只复述教科书结论。可围绕以下问题展开：

1. **模型最依赖哪类词？** 观察是否集中在 `great` / `boring` / `not` 等情感词与否定词。
2. **哪些解释看起来意外或无用？** 例如模型把 `the`、标点等无情感词也赋予了较大归因，说明什么。
3. **否定与混合情绪导致的错误**：`not bad`、`predictable but charming` 这类表达是否让模型翻车。
4. **LSTM vs CNN 对比**（若做了）：哪个更好、为什么——LSTM 擅长长距离依赖与语序，CNN 擅长局部 n-gram 模式且训练更快。
5. **消融实验结论**：替换高贡献词后预测是否如预期改变。

---

## 8. 常见问题与注意事项

- **`torchtext` 接口过时**：本项目已移除 torchtext 依赖，改用自写 `DataLoader`（`training/data.py`），彻底避开新旧版本兼容问题。
- **标签索引**：原参考代码用 `label.data.sub_(1)` 把 1/2 映射为 0/1，且 `label_size=len(label_field.vocab)-1`，建议自己理解后统一标签空间，避免 off-by-one 错误。
- **预训练词向量是可选而非必需**：作业 PDF 未要求。默认随机初始化即可跑通；如需提升，用更小的 GloVe（`glove.6B.100d`，约 340MB）比 3.6GB 的 GoogleNews word2vec 更划算，此时记得把 `embedding_dim` 改为 100。
- **可复现**：随机种子统一在 `utils/seed.py` 里设置；若上 GPU，还需 `torch.cuda.manual_seed_all(seed)` 并设 `torch.backends.cudnn.deterministic=True`。
- **OOV（未登录词）**：词表外词统一映射到 `<unk>`（`UNK_IDX=1`），其 embedding 由随机初始化给出（`np.random.uniform(-0.25, 0.25)`），归因分析时对 OOV 词的解读要谨慎。
