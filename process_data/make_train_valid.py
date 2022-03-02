import pandas as pd
import numpy as np
import collections
import random
from tqdm import tqdm
import argparse
import os
import json

random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--ratio", default=9, type=int,
                        help="#valid over #train.")
parser.add_argument("--neg_count", default=4, type=int,
                        help="negative count.")
args = parser.parse_args()

indexpath = os.path.join(args.dpath, "indexinfo.csv")
train_path = os.path.join(args.dpath, "train.npy")
valid_path = os.path.join(args.dpath, "valid.npy")
test_path = os.path.join(args.dpath, "valid_for_test.npy")
dictionary_path = os.path.join(args.dpath, "model_num.json")
cm_path = os.path.join(args.dpath, "cm.json")

fdf = pd.read_csv(indexpath)
end_index = fdf.shape[0] - 1

cdict = collections.defaultdict(set)
cmdict = {}
cmidx = 0
for pic_index, carm_index in fdf.values.tolist():
    assert(pic_index not in cdict[carm_index]) 
    cdict[carm_index].add(pic_index)
    if carm_index not in cmdict:
        cmdict[carm_index] = cmidx
        cmidx += 1

print(len(cmdict), cmidx)

train_set = []
valid_set = []

for carm_index, pic_set in tqdm(cdict.items(), total=len(cdict), desc='make train and valid'):
    pic_list = list(pic_set)
    cur_len = len(pic_list)
    valid_num = cur_len // args.ratio
    train_num = cur_len - valid_num 
    valid_list = pic_list[:valid_num]
    train_list = pic_list[valid_num:]
    
    for pic_id in train_list:
        train_set.append([pic_id, carm_index])
    for pic_id in valid_list:
        valid_set.append([pic_id, carm_index])


train_set = np.array(train_set)
valid_set = np.array(valid_set)
print(train_set.shape, valid_set.shape)

np.save(train_path, train_set)
np.save(valid_path, valid_set)

saved_index = {int(k): [int(x) for x in v] for k, v in cdict.items()}
json.dump(saved_index, open(dictionary_path, 'w'))
json.dump(cmdict, open(cm_path, 'w'))




