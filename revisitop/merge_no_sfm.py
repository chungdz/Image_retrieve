from PIL import Image
import argparse
import os
from tqdm import tqdm
import pickle
import json
import cv2
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="landmark", type=str,
                        help="root path of all data")
args = parser.parse_args()

train_ox = pd.read_csv(os.path.join(args.dpath, 'oxford5k_train.csv'))
train_pa = pd.read_csv(os.path.join(args.dpath, 'paris6k_train.csv'))

class_idx = 0
img_idx = 0
cdict = {}

for img_c in train_ox['Class'].values:
    if img_c not in cdict:
        cdict[img_c] = class_idx
        class_idx += 1

for img_c in train_pa['Class'].values:
    if img_c not in cdict:
        cdict[img_c] = class_idx
        class_idx += 1

new_list = []
for _, row in train_ox.iterrows():
    cur_row = []
    cur_row.append(img_idx)
    cur_row.append(row['img_name'])
    cur_row.append(cdict[row['Class']])
    cur_row.append(row['img_path'])
    new_list.append(cur_row)
    img_idx += 1

for _, row in train_pa.iterrows():
    cur_row = []
    cur_row.append(img_idx)
    cur_row.append(row['img_name'])
    cur_row.append(cdict[row['Class']])
    cur_row.append(row['img_path'])
    new_list.append(cur_row)
    img_idx += 1

traindf = pd.DataFrame(new_list, columns=['Index', 'img_name', 'Class', 'img_path'])
traindf.to_csv(os.path.join(args.dpath, 'revisit_info.csv'), index=None)

dbm_ox = np.load(os.path.join(args.dpath, 'oxford5k_dbm.npy'))
dbm_pa = np.load(os.path.join(args.dpath, 'paris6k_dbm.npy'))
newnp = np.concatenate([dbm_ox, dbm_pa], axis=0)
np.save(os.path.join(args.dpath, 'revisit_trainset.npy'), newnp)

print(train_ox.shape, train_pa.shape, traindf.shape)
print(dbm_ox.shape, dbm_pa.shape, newnp.shape)


