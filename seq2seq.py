# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 01:31:59 2025

@author: krith
"""

from nltk import download
download('punkt') 
import torch
import torch.nn as nn
import torch.optim as optim
import random

import pandas as pd

df = pd.read_csv("C:/Users/krith/Downloads/seq2seq.csv",encoding="latin-1")
print(df)
pairs = list(zip(df["Source"], df["Target"]))
print(pairs)

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Dummy dataset
'''
pairs = [
    ("i am a student", "je suis un étudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is happy", "elle est heureuse"),
    ("they are playing", "ils jouent"),
    ("you are smart", "tu es intelligent")
];'''

# Tokenize and build vocab
def tokenize(sentence):
    return sentence.lower().split()

def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for sent in sentences:
        for word in tokenize(sent):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_vocab = build_vocab([src for src, tgt in pairs])
tgt_vocab = build_vocab([tgt for src, tgt in pairs])
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

def encode(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in tokenize(sentence)] + [vocab["<eos>"]]

data = [(encode(src, src_vocab), encode(tgt, tgt_vocab)) for src, tgt in pairs]

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMBED_SIZE = 32
HIDDEN_SIZE = 64

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)    # IF U R USING RNN ---> self.rnn = nn.RNN(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

        '''# RNN version
        outputs, hidden = self.rnn(embedded)
        return hidden, None'''

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)    # IF U R USING RNN ----> self.rnn = nn.RNN(emb_dim, hid_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

        ''' # RNN version
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, None'''

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[0]
        batch_size = 1
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(tgt_len, tgt_vocab_size)
        hidden, cell = self.encoder(src)

        input = tgt[0]  # <sos>
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

encoder = Encoder(SRC_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(TGT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
for epoch in range(100):
    total_loss = 0
    for src, tgt in data:
        src_tensor = torch.tensor(src).unsqueeze(1)  # (seq_len, 1)
        tgt_tensor = torch.tensor(tgt).unsqueeze(1)

        optimizer.zero_grad()
        output = model(src_tensor, tgt_tensor)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt_tensor = tgt_tensor[1:].view(-1)

        loss = criterion(output, tgt_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# --------------------------
# Inference
# --------------------------
def translate(model, sentence, max_len=10):
    model.eval()
    tokens = encode(sentence, src_vocab)
    src_tensor = torch.tensor(tokens).unsqueeze(1)

    hidden, cell = model.encoder(src_tensor)
    input = torch.tensor([tgt_vocab["<sos>"]])

    result = []
    for _ in range(max_len):
        output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1).item()
        if top1 == tgt_vocab["<eos>"]:
            break
        result.append(inv_tgt_vocab[top1])
        input = torch.tensor([top1])

    return ' '.join(result)


# Test translation
print("\nTranslation Examples:")
print("Input: 'i am happy'")
print("Output:", translate(model, "i am fine"))

print("Input: 'you are smart'")
print("Output:", translate(model, "thank you"))