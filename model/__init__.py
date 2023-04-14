import sys

sys.path.append(".")
sys.path.append("..")

import model.encoder
import model.decoder
import model.transformer

Encoder = model.encoder.Encoder
Decoder = model.decoder.Decoder
Transformer = model.transformer.Transformer