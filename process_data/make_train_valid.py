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
parser.add_argument("--ratio", default=7, type=int,
                        help="#valid over #train.")
parser.add_argument("--neg_count", default=4, type=int,
                        help="negative count.")
parser.add_argument("--min_len", default=58, type=int, help="min length of data")
args = parser.parse_args()

indexpath = os.path.join(args.dpath, "indexinfo.csv")
train_path = os.path.join(args.dpath, "train.npy")
train_path2 = os.path.join(args.dpath, "train2.npy")
valid_path = os.path.join(args.dpath, "valid.npy")
test_path = os.path.join(args.dpath, "valid_for_test.npy")
dictionary_path = os.path.join(args.dpath, "model_num.json")
discp = os.path.join(args.dpath, "cutted_class.json")
discarded = []

fdf = pd.read_csv(indexpath)
end_index = fdf.shape[0] - 1

cdict = collections.defaultdict(set)
for pic_index, carm_index in fdf.values:
    assert(pic_index not in cdict[carm_index]) 
    cdict[carm_index].add(pic_index)

train_set = []
train_set_reverse = []
valid_set = []
test_set = []
all_valid = set()
for carm_index, pic_set in tqdm(cdict.items(), total=len(cdict), desc='make valid index set'):
    pic_list = list(pic_set)
    cur_len = len(pic_list)
    valid_num = cur_len // args.ratio
    valid_list = pic_list[:valid_num]
    for vidx in valid_list:
        all_valid.add(vidx)

for carm_index, pic_set in tqdm(cdict.items(), total=len(cdict), desc='make train and valid'):
    pic_list = list(pic_set)
    cur_len = len(pic_list)
    valid_num = cur_len // args.ratio
    train_num = cur_len - valid_num 
    valid_list = pic_list[:valid_num]
    train_list = pic_list[valid_num:]
    
    for i in range(train_num - 1):
        for j in range(i + 1, train_num):
            new_sample = [train_list[i], train_list[j]]
            while len(new_sample) < 2 + args.neg_count:
                neg_idx = random.randint(0, end_index - 1)
                if neg_idx not in pic_set and neg_idx not in new_sample and neg_idx not in all_valid:
                    new_sample.append(neg_idx)
            train_set.append(new_sample)

            new2 = [train_list[j], train_list[i]]
            while len(new2) < 2 + args.neg_count:
                neg_idx = random.randint(0, end_index - 1)
                if neg_idx not in pic_set and neg_idx not in new2 and neg_idx not in all_valid:
                    new2.append(neg_idx)
            train_set_reverse.append(new2)
    
    if len(pic_set) < args.min_len:
        discarded.append(carm_index)
        continue
    
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
    
    for pidx in valid_list:
        test_set.append([pidx, carm_index])

train_set = np.array(train_set)
train_set_reverse = np.array(train_set_reverse)
valid_set = np.array(valid_set)
test_set = np.array(test_set)
print(train_set.shape, train_set_reverse.shape, valid_set.shape, test_set.shape)
print(len(discarded), "class was discarded")

np.save(train_path, train_set)
np.save(train_path2, train_set_reverse)
np.save(valid_path, valid_set)
np.save(test_path, test_set)

saved_index = {int(k): [int(x) for x in v] for k, v in cdict.items()}
json.dump(saved_index, open(dictionary_path, 'w'))
json.dump(discarded, open(discp, 'w'))




