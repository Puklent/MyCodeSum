import sys

sys.path.append(".")
sys.path.append("..")

import csv
import json
import data.dataprocess
import torch.utils.data as Data

# Transformer parameters:
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

sentences = []
with open("/data/alumpuk/github/TransCodeSum/data/train/train_sbt.token.code", "r") as sbt_file:
    sbts = sbt_file.readlines()
with open("/data/alumpuk/github/TransCodeSum/data/train/train_tmp.token.nl", "r") as nl_file:
    nls = nl_file.readlines()
for i in range(len(nls)):
    sentence = []
    sentence.append(sbts[i])
    sentence.append(nls[i])
    sentence.append(nls[i])
    sentences.append(sentence)

with open("/data/alumpuk/github/TransCodeSum/data/vocab/src_vocab.json", "r") as src_vocab_file:
    src_vocab = json.load(src_vocab_file)

with open("/data/alumpuk/github/TransCodeSum/data/vocab/tgt_vocab.json", "r") as tgt_vocab_file:
    tgt_vocab = json.load(tgt_vocab_file)

with open("/data/alumpuk/github/TransCodeSum/data/vocab/idx2word.json", "r") as idx2word_file:
    idx2word = json.load(idx2word_file)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
src_len = 512
tgt_len = 128

sbt_file.close()
nl_file.close()
src_vocab_file.close()
tgt_vocab_file.close()
idx2word_file.close()

enc_inputs, dec_inputs, dec_outputs = data.dataprocess.make_data(sentence, src_vocab, tgt_vocab)
loader = Data.DataLoader(data.dataprocess.MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

if __name__ == '__main__':
    print(sentence)