from __future__ import print_function
import math
import torch
import torch.nn as nn
from TDNN.tdnn import TDNN

class XVector(nn.Module):
    def __init__(self, nfea, nhid, nframeout, nsegout, nspeaker):
        super(XVector, self).__init__()
        self.frame1 = TDNN(input_dim=nfea, output_dim=nhid, context_size=5, dilation=1, batch_norm=False)
        self.frame2 = TDNN(input_dim=nhid, output_dim=nhid, context_size=3, dilation=2, batch_norm=False)
        self.frame3 = TDNN(input_dim=nhid, output_dim=nhid, context_size=3, dilation=3, batch_norm=False)
        self.frame4 = TDNN(input_dim=nhid, output_dim=nhid, context_size=1, dilation=1, batch_norm=False)
        self.frame5 = TDNN(input_dim=nhid, output_dim=nframeout, context_size=1, dilation=1, batch_norm=False)
        self.segment6 = nn.Linear(nframeout*2, nsegout)
        self.segment7 = nn.Linear(nsegout, nsegout)
        self.decoder = nn.Linear(nsegout, nspeaker)

    def forward(self, framefeaseq):
        '''framefeaseq: [bsize, window_len, feature_size]
        '''
        frame1out = self.frame1(framefeaseq)
        frame2out = self.frame2(frame1out)
        frame3out = self.frame3(frame2out)
        frame4out = self.frame4(frame3out)
        frame5out = self.frame5(frame4out)
        # [bsize, seqlen, nframeout] -> [bsize, nframeout]
        frame_mean = torch.mean(frame5out, dim=1)
        frame_var = torch.var(frame5out, dim=1)
        seglevelfea = torch.cat([frame_mean, frame_var], 1)
        seg6out = nn.functional.relu(self.segment6(seglevelfea))
        seg7out = nn.functional.relu(self.segment7(seg6out))
        output = self.decoder(seg7out)
        return output
