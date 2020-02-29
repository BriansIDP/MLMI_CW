import os
import re
from torch.utils.data import Dataset, DataLoader
import torch

def get_scp_labels(scp, datapath):
    """Re-arrange the data into utterance files and save in each SCP entry"""
    scp_dict = {}
    newscps = {}
    with open(scp, 'r') as fin:
        scplines = fin.readlines()
    for line in scplines:
        scpname, scpaddress = line.strip().split('=')
        scpname = scpname.strip('.fbk')
        scpidx = re.findall('[0-9]*,[0-9]*', scpaddress)[0]
        start, end = scpidx.split(',')
        scpfilename = re.findall('AMI_[a-zA-Z0-9]*_MDM.fbk', scpaddress)[0]
        if scpfilename not in scp_dict:
            scp_dict[scpfilename] = [(scpname, start, end)]
        else:
            scp_dict[scpfilename].append((scpname, start, end))
    for filename, elems in scp_dict.items():
        print(filename)
        with open(os.path.join(datapath, filename)) as fin:
            lines = fin.readlines()[1:-1]
        features = []
        featurebuffer = []
        for line in lines:
            numbers = line.split()
            if ':' in numbers[0]:
                features.append(' '.join(featurebuffer) + '\n')
                featurebuffer = []
                featurebuffer += numbers[1:]
            else:
                featurebuffer += numbers
        features.append(featurebuffer)
        for scpname, start, end in elems:
            filepath = os.path.join(datapath, 'fbk', scpname)
            with open(os.path.join(datapath, 'fbk', scpname), 'w') as fout:
                lines = features[int(start):int(end)+1]
                fout.writelines(lines)
            print(filepath)

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
        self.files = []
        lab_idx = 0
        with open(scpmap, 'r') as fin:
            for line in fin:
                scp, lab = line.strip().split()
                _, _, start, end = scp.split('_')
                windows = segment_into_windows(int(start), int(end), seglen, overlap)
                for window in windows:
                    self.files.append([scp, labdict.speaker2idx[lab], window])
        self.path = fbkpath
        self.seglen = seglen
        self.overlap = overlap

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        scp, lab, window = self.files[idx]
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
        return features, lab

def collate_fn(batch):
    fbk, lab = list(zip(*batch))
    fbks = torch.cat(fbk, 0)
    labs = torch.LongTensor(lab)
    return fbks, labs

def create(fbkpath, scppath, seglen=200, overlap=100, batchSize=1, shuffle=False, workers=0):
    loaders = []
    speakerfile = os.path.join(scppath, 'speakers')
    labdict = SpeakerList(speakerfile)
    for split in ['train', 'valid']:
        scpmap = os.path.join(scppath, split + '_map.scp')
        dataset = FBankdata(scpmap, fbkpath, seglen, overlap, labdict)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
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
