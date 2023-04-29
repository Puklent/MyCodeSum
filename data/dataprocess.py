import torch.nn
import torch.utils.data as Data

src_len = 512
tgt_len = 128

def make_data(sentences, src_vocab, tgt_vocab):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input, dec_input, dec_output = [], [], []
        for n in sentences[i][0].split():
            if n in src_vocab.keys():
                enc_input.append(src_vocab[n])
            else:
                enc_input.append(src_vocab['<UNK>'])
        while len(enc_input) < src_len:
            enc_input.append(src_vocab['<NONE>'])

        for n in sentences[i][1].split():
            if n in tgt_vocab.keys():
                dec_input.append(tgt_vocab[n])
            else:
                dec_input.append(tgt_vocab['<UNK>'])
        while len(dec_input) < tgt_len:
            dec_input.append(tgt_vocab['<NONE>'])

        for n in sentences[i][2].split():
            if n in tgt_vocab.keys():
                dec_output.append(tgt_vocab[n])
            else:
                dec_output.append(tgt_vocab['<UNK>'])
        while len(dec_output) < tgt_len:
            dec_output.append(tgt_vocab['<NONE>'])

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

    