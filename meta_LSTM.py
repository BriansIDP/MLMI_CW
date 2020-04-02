from __future__ import print_function
import torch.nn as nn
import torch
from torch.autograd import Variable


def LSTMCell(input, hidden, parameters):
    hx, cx = hidden
    w_ih, w_hh, b_ih, b_hh = parameters
    gates = nn.functional.linear(input, w_ih, b_ih) + nn.functional.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy

class LSTMemb(nn.Module):
    def __init__(self, ninp, nhid, ninterm, nlayers, nspeaker, rnndrop=0.5, dropout=0.5):
        super(LSTMemb, self).__init__()
        self.LSTM_stack = nn.LSTM(ninp, nhid, num_layers=nlayers, batch_first=True, dropout=rnndrop, bidirectional=False)
        # self.LSTM_copy_fast = nn.LSTM(ninp, nhid, num_layers=nlayers, batch_first=True, dropout=rnndrop, bidirectional=False)
        self.projection = nn.Linear(nhid, ninterm)
        self.drop = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(ninterm, nspeaker)
        self.init_weights()

        self.nlayers = nlayers
        self.nhid = nhid

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

    def forward_fast_weights(self, features, fast_weights):
        # Get all fast weights assigned
        functional_lstm_params = []
        for layer in range(self.nlayers):
            w_ih = fast_weights['LSTM_stack.weight_ih_l'+str(layer)]
            w_hh = fast_weights['LSTM_stack.weight_hh_l'+str(layer)]
            b_ih = fast_weights['LSTM_stack.bias_ih_l'+str(layer)]
            b_hh = fast_weights['LSTM_stack.bias_hh_l'+str(layer)]
            functional_lstm_params.append([w_ih, w_hh, b_ih, b_hh])
        decoder_weight = fast_weights['decoder.weight']
        decoder_bias = fast_weights['decoder.bias']
        projection_weight = fast_weights['projection.weight']
        projection_bias = fast_weights['projection.bias']
        # Forward
        bsz = features.size(0)
        nframes = features.size(1)
        hidden = self.init_hidden(bsz)
        layer_output = features
        for layer in range(self.nlayers):
            layer_param = functional_lstm_params[layer]
            layer_output_list = []
            for i in range(nframes):
                hidden = LSTMCell(layer_output[:,i,:], hidden, layer_param)
                layer_output_list.append(hidden[0].unsqueeze(1))
            layer_output = torch.cat(layer_output_list, 1)
        z1 = layer_output[:,nframes-1,:]
        z1 = nn.functional.linear(z1, projection_weight, projection_bias)
        z1 = nn.functional.relu(z1)
        z1 = nn.functional.linear(z1, decoder_weight, decoder_bias)
        return z1

    def copy_model_weights(self):
        fast_weights = {}
        for name, pr in self.named_parameters():
            fast_weights[name] = pr.clone()
        return fast_weights

    def update_fast_grad(self, fast_weights, names, fast_grad, lr_a):
        if len(fast_weights) != len(fast_grad): raise ValueError('fast grad parameter number not match')
        num = len(fast_grad)
        updated_fast_params = {}
        for i in range(num):
            updated_fast_params[names[i]] = (fast_weights[i] - lr_a * fast_grad[i])
        return updated_fast_params

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid))
