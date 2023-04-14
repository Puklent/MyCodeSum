import sys

sys.path.append(".")
sys.path.append("..")

import math
import torch
import torch.nn as nn
from bin.var import src_vocab_size

"""
    Complete the task of position embedding.
    parameters:
        d_model: The dimension of a word vector.
        dropout: The rate of temporarily dropping out some nodes during training.
        max_len: The max length of one sample. That is the length of the sentence
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super().__init__()
    
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Module Test.

if __name__ == '__main__':
    import csv
    d_model = 512  # Embedding Size
    d_ff = 2048 # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    # with open('Transformer\data\\tmp_data.csv', 'r', encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)
    src_emb = nn.Embedding(src_vocab_size, d_model)
    pos_emb = PositionalEncoding(d_model)