from __future__ import print_function
import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTMemb(nn.Module):
    def __init__(self, ninp, nhid, ninterm, nlayers, nspeaker, rnndrop=0.5, dropout=0.5):
        super(LSTMemb, self).__init__()
        self.LSTM_stack = nn.LSTM(ninp, nhid, num_layers=nlayers, batch_first=True, dropout=rnndrop, bidirectional=False)
        self.projection = nn.Linear(nhid, ninterm)
        self.drop = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(ninterm, nspeaker)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, features):
        nframes = features.size(1)
        # [batch, nframes, n_mel]
        LSTMout, hidden = self.LSTM_stack(features)
        LSTMout = LSTMout[:,nframes-1,:]
        projected = nn.functional.relu(self.projection(LSTMout))
        projected = self.drop(projected)
        output = self.decoder(projected)
        return output
