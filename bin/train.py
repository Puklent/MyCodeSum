import sys

sys.path.append(".")
sys.path.append("..")

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import model
from bin import test
from bin.var import loader, tgt_vocab, idx2word

transformer = model.transformer.Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

def train():
    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    enc_inputs, _, _ = next(iter(loader))
    enc_inputs = enc_inputs.cuda()
    for i in range(len(enc_inputs)):
        greedy_dec_input = test.greedy_decoder(transformer, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"], tgt_vocab=tgt_vocab)
        predict, _, _, _ = transformer(enc_inputs[i].view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])

if __name__ == '__main__':
    train()