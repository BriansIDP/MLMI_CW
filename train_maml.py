import argparse
import time
import math
import os
import torch
import torch.nn as nn
from operator import itemgetter

import meta_dataloader
from model import XVector
from meta_LSTM import LSTMemb

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
parser.add_argument('--bsize', type=int, default=200,
                    help='batch size')
parser.add_argument('--nfea', type=int, default=41,
                    help='size of the FBank features')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden state size')
parser.add_argument('--nframeout', type=int, default=1500,
                    help='frame-level output size, before stats pool')
parser.add_argument('--nsegout', type=int, default=512,
                    help='segment-level output size')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers to be used for LSTM')
parser.add_argument('--lr', type=float, default=10.0,
                    help='learning rate')
parser.add_argument('--iterations', type=int, default=60000,
                    help='upper epoch limit')
parser.add_argument('--log-interval', type=int, default=20,
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
parser.add_argument('--nlabels', type=int, default=5,
                    help='number of speakers in each task')
parser.add_argument('--npoints', type=int, default=10,
                    help='number of samples for each speaker')
parser.add_argument('--ntasks', type=int, default=16,
                    help='number of tasks for each meta update')
parser.add_argument('--nsteps', type=int, default=1,
                    help='number of tasks for each meta update')
parser.add_argument('--lr_a', type=float, default=0.2,
                    help='fast weight learning rate')
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
train_loader, val_loader, labdict = meta_dataloader.create(args.data, args.label_scp, seglen=args.window_size, overlap=args.overlap, shuffle=args.shuffle, batchSize=args.bsize)
nspeaker = labdict.get_n_speakers()

###############################################################################
# Build the model
###############################################################################
if args.model == 'TDNN':
    model = XVector(args.nfea, args.nhid, args.nframeout, args.nsegout, nspeaker)
elif args.model == 'LSTM':
    model = LSTMemb(args.nfea, args.nhid, args.nsegout, args.nlayers, args.nlabels)
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

def train(data, model, iteration, lr):
    meta_learning_loss = 0.0
    start_time = time.time()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8)
    task_list = []
    meta_task_list = []
    for i in range(args.ntasks):
        fast_task, meta_task = data.get_task_subset(args.nlabels, args.npoints)
        task_list.append(fast_task)
        meta_task_list.append(meta_task)
    meta_learning_loss = 0
    for i, task in enumerate(task_list):
        # Copy current model weights to fast_weights
        fast_weights = model.copy_model_weights()
        # Arrange the current batch
        features, labels = list(zip(*task))
        features_in = torch.cat(features, 0).to(device)
        labels = torch.tensor(labels).to(device)
        # Fast weight iterations
        for grad_update_iter in range(args.nsteps):
            output = model.forward_fast_weights(features_in, fast_weights)
            names, weights = list(zip(*list(fast_weights.items())))
            train_loss = criterion(output, labels)
            fast_grad = torch.autograd.grad(train_loss, weights, create_graph=True)
            fast_weights = model.update_fast_grad(weights, names, fast_grad, args.lr_a)

        # Do meta forwarding
        meta_task = meta_task_list[i]
        # Arrange the current batch
        features, labels = list(zip(*meta_task))
        features_in = torch.cat(features, 0).to(device)
        labels = torch.tensor(labels).to(device)
        output = model.forward_fast_weights(features_in, fast_weights)
        meta_learning_loss += criterion(output, labels)

    # Outer loop updates
    meta_learning_loss /= args.ntasks
    meta_learning_loss.backward()
    optimiser.step()
    model.zero_grad()

    return model, meta_learning_loss.item()

###############################################################################
# Main Code
###############################################################################
logging('Training Start!')
for pairs in arglist:
    logging(pairs[0] + ': ' + str(pairs[1]))
lr = args.lr
total_meta_loss = 0.
best_meta_loss = None
try:
    for iteration in range(1, args.iterations+1):
        start = time.time()
        # Train model
        model, meta_loss = train(train_loader, model, iteration, lr)
        total_meta_loss += meta_loss
        if iteration % args.log_interval == 0 and iteration > 0:
            elapsed = time.time() - start
            logging('| {:5d}/{:5d} iterations | lr {:02.5f} | ms/iteration {:5.2f} | '
                'loss {:5.2f}'.format(
            iteration, args.iterations, lr, 
            elapsed * 1000 / args.log_interval, total_meta_loss / args.log_interval))

            if not best_meta_loss or total_meta_loss < best_meta_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_meta_loss = total_meta_loss
            else:
                lr = lr / 1.00
            total_meta_loss = 0.
            start = time.time()

except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_start_time = time.time() 
