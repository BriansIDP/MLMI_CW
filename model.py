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
        frameout = self.frame1(framefeaseq)
        frameout = self.frame2(frameout)
        frameout = self.frame3(frameout)
        frameout = self.frame4(frameout)
        frameout = self.frame5(frameout)
        # [bsize, seqlen, nframeout] -> [bsize, nframeout]
        frame_mean = torch.mean(frameout, dim=1)
        frame_var = torch.var(frameout, dim=1)
        seglevelfea = torch.cat([frame_mean, frame_var], 1)
        segout = nn.functional.relu(self.segment6(seglevelfea))
        segout = nn.functional.relu(self.segment7(segout))
        output = self.decoder(segout)
        return output
