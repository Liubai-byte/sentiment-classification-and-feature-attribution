import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size,
                 dropout=0.5, num_filters=100, filter_sizes=(3, 4, 5)):
        super(CNNSentiment, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Multi-size kernels capture n-gram local patterns of different lengths
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes]
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(num_filters * len(filter_sizes), label_size)

    def init_hidden(self):
        # CNN has no recurrent state; return None to keep a unified interface with the LSTM training loop
        return None

    def forward(self, sentence):
        # sentence: (seq_len, batch)
        x = self.embeddings(sentence).permute(1, 2, 0)          # (batch, dim, seq_len)
        x = [F.relu(conv(x)) for conv in self.convs]            # each (batch, filters, *)
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]  # (batch, filters)
        x = self.dropout_layer(torch.cat(x, dim=1))             # (batch, filters * num_sizes)
        y = self.hidden2label(x)                                # (batch, label_size)
        return F.log_softmax(y, dim=1)
