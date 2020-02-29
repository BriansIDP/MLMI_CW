with open('align_fname_spkrname') as fin:
    lines = fin.readlines()

train = []
valid = []
for i, line in enumerate(lines):
    if i != 0 and i % 10 == 0:
        valid.append(line)
    else:
        train.append(line)

with open('train_map.scp', 'w') as fout:
    fout.writelines(train)
with open('valid_map.scp', 'w') as fout:
    fout.writelines(valid)
