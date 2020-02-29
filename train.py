import argparse
import time
import math
import os
import torch
import torch.nn as nn
from operator import itemgetter

import data_utils
from model import XVector
from LSTM import LSTMemb

arglist = []
parser = argparse.ArgumentParser(description='PyTorch X-vector system on AMI')
parser.add_argument('--data', type=str, default='./data/fbk',
                    help='location of the filterbank features')
parser.add_argument('--label_scp', type=str, default='./lib/mlabs/',
                    help='location of the utterance level file list with labels')
parser.add_argument('--window_size', type=int, default=200,
                    help='No. of frames in a window')
parser.add_argument('--overlap', type=int, default=100,
                    help='No. of frames overlapped between adjacent windows')
parser.add_argument('--shuffle', action='store_true',
                    help='whether to shuffle the data at window level or not')
parser.add_argument('--cuda', action='store_true',
                    help='whether to use GPU or not')
parser.add_argument('--bsize', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--nfea', type=int, default=41, metavar='N',
                    help='size of the FBank features')
parser.add_argument('--nhid', type=int, default=512, metavar='N',
                    help='hidden state size')
parser.add_argument('--nframeout', type=int, default=1500, metavar='N',
                    help='frame-level output size, before stats pool')
parser.add_argument('--nsegout', type=int, default=512, metavar='N',
                    help='segment-level output size')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers to be used for LSTM')
parser.add_argument('--lr', type=float, default=10.0,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to embeddings (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--seed', type=int, default=999,
                    help='manual random seed')
parser.add_argument('--save', type=str, default='models/model.pt',
                    help='path to save the model')
parser.add_argument('--logfile', type=str, default='LOG/xvec.log',
                    help='path to the log file')
parser.add_argument('--model', type=str, choices=['TDNN', 'LSTM'], default='LSTM',
                    help='type of speaker embedding model to use')
args = parser.parse_args()

arglist.append(('Data', args.data))
arglist.append(('Label and SCP', args.label_scp))

def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
train_loader, val_loader, labdict = data_utils.create(args.data, args.label_scp, seglen=args.window_size, overlap=args.overlap, shuffle=args.shuffle, batchSize=args.bsize, workers=0)
nspeaker = labdict.get_n_speakers()

###############################################################################
# Build the model
###############################################################################
if args.model == 'TDNN':
    model = XVector(args.nfea, args.nhid, args.nframeout, args.nsegout, nspeaker)
elif args.model == 'LSTM':
    model = LSTMemb(args.nfea, args.nhid, args.nsegout, args.nlayers, nspeaker)
if args.cuda:
    model.cuda()

###############################################################################
# Criterion
###############################################################################
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training and Evaluation code
###############################################################################

def evaluate(data, model):
    total_loss = 0.0
    nbatches = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, databatch in enumerate(data):
            features, labels = databatch
            output = model(features.to(device))
            loss = criterion(output, labels.to(device))
            total_loss += loss.item()
            nbatches += labels.size(0)
    total_loss = total_loss / nbatches
    return total_loss

def train(data, model, epoch, lr):
    total_loss = 0.0
    start_time = time.time()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
    for batch, databatch in enumerate(data):
        features, labels = databatch
        output = model(features.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimiser.step()
        total_loss += loss.item()

        # For logging
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(data), lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
    return model

###############################################################################
# Main Code
###############################################################################
logging('Training Start!')
for pairs in arglist:
    logging(pairs[0] + ': ' + str(pairs[1]))
best_val_loss = None
lr = args.lr
try:
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Train model
        model = train(train_loader, model, epoch, lr)
        val_loss = evaluate(val_loader, model)
        logging('-' * 89)
        logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch+1, time.time() - epoch_start, val_loss))
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 2.0

except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_start_time = time.time() 
