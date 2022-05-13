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

train_sfm = pd.read_csv(os.path.join(args.dpath, 'train_info.csv'))
train_ox = pd.read_csv(os.path.join(args.dpath, 'oxford5k_train.csv'))
train_pa = pd.read_csv(os.path.join(args.dpath, 'paris6k_train.csv'))

class_idx = train_sfm['Class'].max() + 1
img_idx = train_sfm['Index'].max() + 1
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
final_df = pd.concat([train_sfm, traindf], axis=0)
final_df.to_csv(os.path.join(args.dpath, 'train_info_final.csv'), index=None)

dbm_ox = np.load(os.path.join(args.dpath, 'oxford5k_dbm.npy'))
dbm_pa = np.load(os.path.join(args.dpath, 'paris6k_dbm.npy'))
newnp = np.concatenate([dbm_ox, dbm_pa], axis=0)
np.save(os.path.join(args.dpath, 'trainset', 'train_image_set5.npy'), newnp)

print(train_sfm.shape, train_ox.shape, train_pa.shape, final_df.shape)
print(dbm_ox.shape, dbm_pa.shape, newnp.shape)


