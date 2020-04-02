import os
import re
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch

random.seed(1)

def segment_into_windows(start, end, seglen=200, overlap=100):
    current_f = 0
    window_list = []
    duration = end - start
    while current_f + seglen < duration:
        window_list.append([current_f, current_f+seglen])
        current_f = current_f + seglen - overlap
    window_list.append([current_f, duration])
    return window_list


class SpeakerList():
    def __init__(self, spk_label_file):
        self.speaker2idx = {}
        self.idx2speaker = []
        with open(spk_label_file) as fin:
            for line in fin:
                speaker = line.strip()
                self.speaker2idx[speaker] = len(self.idx2speaker)
                self.idx2speaker.append(speaker)

    def get_n_speakers(self):
        return len(self.idx2speaker)


class FBankdata(Dataset):
    def __init__(self, scpmap, fbkpath, seglen, overlap, labdict):
        """Read in SCP
        """
        self.task_pool = {}
        lab_idx = 0
        with open(scpmap, 'r') as fin:
            for line in fin:
                scp, lab = line.strip().split()
                _, _, start, end = scp.split('_')
                windows = segment_into_windows(int(start), int(end), seglen, overlap)
                for window in windows:
                    spk_id = labdict.speaker2idx[lab]
                    if spk_id in self.task_pool:
                        self.task_pool[spk_id].append([scp, window])
                    else:
                        self.task_pool[spk_id] = [[scp, window]]
        self.tasks = list(self.task_pool.keys())
        self.num_tasks = len(self.tasks)
        self.path = fbkpath
        self.seglen = seglen
        self.overlap = overlap

    def __len__(self):
        return len(self.files)

    def getitem(self, scp, window):
        fbkfilename = os.path.join(self.path, scp)
        features = []
        with open(fbkfilename, 'r') as fin:
            lines = fin.readlines()
            n = len(lines)
        start, end = window
        for line in lines[start:end]:
            features.append(torch.tensor([float(f) for f in line.strip().split()]))
        ninp = features[0].size(0)
        if len(features) < self.seglen:
            features += [torch.zeros(ninp)] * (self.seglen - len(features))
        features = torch.cat(features, 0).view(1, self.seglen, ninp)
        return features

    def get_task_subset(self, ntasks, npoints):
        minibatch = []
        metabatch = []
        speaker_inds = random.choices(self.tasks, k=ntasks)
        for ids, i in enumerate(speaker_inds):
            task_samples = random.choices(self.task_pool[i], k=npoints)
            for sample in task_samples:
                scp, window = sample
                features = self.getitem(scp, window)
                minibatch.append([features, ids])
            meta_samples = random.choices(self.task_pool[i], k=npoints)
            for sample in task_samples:
                scp, window = sample
                features = self.getitem(scp, window)
                metabatch.append([features, ids])
        return minibatch, metabatch

    def get_eval_task_subset(self, nspeaker, npoints, id_start):
        minibatch = []
        metabatch = []
        speaker_inds = self.tasks[id_start:id_start+nspeaker]
        for ids, i in enumerate(speaker_inds):
            task_samples = random.choices(self.task_pool[i], k=npoints)
            for sample in task_samples:
                scp, window = sample
                features = self.getitem(scp, window)
                minibatch.append([features, ids])
            meta_samples = random.choices(self.task_pool[i], k=npoints)
            for sample in task_samples:
                scp, window = sample
                features = self.getitem(scp, window)
                metabatch.append([features, ids])
        return minibatch, metabatch, id_start+nspeaker


def collate_fn(batch):
    fbk, lab = list(zip(*batch))
    fbks = torch.cat(fbk, 0)
    labs = torch.LongTensor(lab)
    return fbks, labs

def create(fbkpath, scppath, seglen=200, overlap=100, batchSize=1, shuffle=False, workers=0):
    loaders = []
    speakerfile = os.path.join(scppath, 'speakers')
    labdict = SpeakerList(speakerfile)
    for split in ['train', 'test']:
        scpmap = os.path.join(scppath, split + '_map.scp')
        dataset = FBankdata(scpmap, fbkpath, seglen, overlap, labdict)
        loaders.append(dataset)
        # loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
        #                           shuffle=shuffle, collate_fn=collate_fn,
        #                           num_workers=workers))
    return loaders[0], loaders[1], labdict


if __name__ == "__main__":
    # scp = "/home/dawna/gs534/Documents/Project/exp/MLMI4/lib/traincv.scp"
    label = "/home/dawna/gs534/Documents/Project/exp/MLMI4/lib/mlabs/"
    datapath = "/home/dawna/gs534/Documents/Project/exp/MLMI4/data/fbk"
    # scpdict = get_scp_labels(scp, datapath)
    train_data, valid_data = create(datapath, label, batchSize=3, workers=0)
    for databatch in train_data:
        import pdb; pdb.set_trace()
        print(databatch)
