import pandas as pd
import numpy as np
import collections
import random
from tqdm import tqdm
import argparse
import os
import json

random.seed(7)

radio = 7
neg_count = 4

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--ratio", default=7, type=int,
                        help="#valid over #train.")
parser.add_argument("--neg_count", default=4, type=int,
                        help="negative count.")
args = parser.parse_args()

indexpath = os.path.join(args.dpath, "indexinfo.csv")
train_path = os.path.join(args.dpath, "train.npy")
valid_path = os.path.join(args.dpath, "valid.npy")

fdf = pd.read_csv(indexpath)
end_index = fdf.shape[0] - 1

cdict = collections.defaultdict(set)
for pic_index, carm_index in fdf.values:
    assert(pic_index not in cdict[carm_index]) 
    cdict[carm_index].add(pic_index)

train_set = []
valid_set = []
for carm_index, pic_set in tqdm(cdict.items(), total=len(cdict), desc='make train and valid'):
    pic_list = list(pic_set)
    cur_len = len(pic_list)
    valid_num = cur_len // args.ratio
    train_num = cur_len - valid_num 
    valid_list = pic_list[:valid_num]
    train_list = pic_list[valid_num:]
    
    for i in range(train_num - 1):
        for j in range(i + 1, train_num):
            new_sample = [pic_list[i], pic_list[j]]
            while len(new_sample) < 2 + args.neg_count:
                neg_idx = random.randint(0, end_index - 1)
                if neg_idx not in pic_set and neg_idx not in new_sample:
                    new_sample.append(neg_idx)
            train_set.append(new_sample)
    
    pos_sample_valid = random.sample(train_list, valid_num)
    for i in range(valid_num):
        new_sample_pos = [valid_list[i], pos_sample_valid[i], 1]
        new_sample_neg = [valid_list[i]]
        while len(new_sample_neg) < 2:
            neg_idx = random.randint(0, end_index - 1)
            if neg_idx not in pic_set and neg_idx != valid_list[i]:
                new_sample_neg.append(neg_idx)
        new_sample_neg.append(0)
        valid_set.append(new_sample_pos)
        valid_set.append(new_sample_neg)

train_set = np.array(train_set)
valid_set = np.array(valid_set)
print(train_set.shape, valid_set.shape)

np.save(train_path, train_set)
np.save(valid_path, valid_set)

