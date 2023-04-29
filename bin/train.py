import sys

sys.path.append(".")
sys.path.append("..")

import csv
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import model
import data.dataprocess
from bin import test
from bin.var import src_vocab, tgt_vocab, idx2word, enc_inputs, dec_inputs, dec_outputs
import utils.timer
from utils.logger import logger, handler1, handler2

def train():
    epoch_time = utils.timer.Timer()
    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = transformer(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            logger.info('[ train: Epoch %04d | loss = %.6f | Time for epoch: %.2f (s) ]' 
                    % (epoch + 1, loss, epoch_time.time()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
    logger.info('[ End of training, Start testing...                               ]')
    # enc_inputs, _, _ = next(iter(loader))
    # enc_inputs = enc_inputs.cuda()
    # for i in range(len(enc_inputs)):
    #     greedy_dec_input = test.greedy_decoder(transformer, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"], tgt_vocab=tgt_vocab)
    #     predict, _, _, _ = transformer(enc_inputs[i].view(1, -1), greedy_dec_input)
    #     predict = predict.data.max(1, keepdim=True)[1]
    #     print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])

if __name__ == '__main__':
    # load and train
    logger.info('[ ------------------------------Main----------------------------- ]')
    logger.info('[ Load and process data files...                                  ]')
    
    loader = Data.DataLoader(data.dataprocess.MyDataSet(enc_inputs, dec_inputs, dec_outputs), 16, True)
    transformer = model.transformer.Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

    logger.info('[ All preparation OK, start training...                           ]')
    train()