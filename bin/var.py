import sys

sys.path.append(".")
sys.path.append("..")

import csv
import data.dataprocess
import torch.utils.data as Data

# Transformer parameters:
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


# reading data
sentence, src_vocab, tgt_vocab = [], {}, {}
with open('../data/original.csv', 'r', encoding="utf-8") as original:
    reader = csv.reader(original)
    for line in reader:
        sentence.append(line)

with open('../data/src_mapping.csv', 'r', encoding="utf-8") as src_mapping:
    reader = csv.reader(src_mapping)
    for line in reader:
        src_vocab[line[0]] = line[1]

with open('../data/tgt_mapping.csv', 'r', encoding="utf-8") as tgt_mapping:
    reader = csv.reader(tgt_mapping)
    for line in reader:
        tgt_vocab[line[0]] = line[1]

original.close()
src_mapping.close()
tgt_mapping.close()

src_vocab_size = len(src_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length

enc_inputs, dec_inputs, dec_outputs = data.dataprocess.make_data(sentence, src_vocab, tgt_vocab)
loader = Data.DataLoader(data.dataprocess.MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)