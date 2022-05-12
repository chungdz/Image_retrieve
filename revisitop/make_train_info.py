import argparse
import os
from tqdm import tqdm
import pickle
import json
import cv2
import numpy as np
import json
import pandas as pd
import gc
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="landmark", type=str,
                        help="root path of all data")
parser.add_argument("--img_size", default=224, type=int,
                        help="image size")
parser.add_argument("--step", default=20000, type=int,
                        help="matrix length")
args = parser.parse_args()

img_path = os.path.join(args.dpath, 'SFM')
save_path = os.path.join(args.dpath, 'trainset')
pdict = os.path.join(args.dpath, 'retrieval-SfM-120k.pkl')
path_dict = os.path.join(args.dpath, 'sfmpath.json')
infopath = os.path.join(args.dpath, 'train_info.csv')
trainp = 'train_image_set'

if os.path.exists(path_dict):
    filep = json.load(open(path_dict, 'r', encoding='utf8'))
else:
    filep = {}
    for root, dirs, files in tqdm(os.walk(img_path)):
        for fname in files:
            filep[fname] = os.path.join(root, fname)
    json.dump(filep, open(path_dict, 'w', encoding='utf8'))

dic_info = pickle.load(open(pdict, 'rb'))
img_list = []
idx = 0
valid_idx = set()
# class minus 1 because class in pdict start from 1
for cids, iclass in tqdm(zip(dic_info['val']['cids'], dic_info['val']['cluster']), total=len(dic_info['val']['cids'])):
    assert(cids not in valid_idx)
    img_list.append([idx, cids, iclass - 1, filep[cids]])
    idx += 1
    valid_idx.add(cids)

for cids, iclass in tqdm(zip(dic_info['train']['cids'], dic_info['train']['cluster']), total=len(dic_info['train']['cids'])):
    assert(cids not in valid_idx)
    img_list.append([idx, cids, iclass - 1, filep[cids]])
    idx += 1
    valid_idx.add(cids)

infodf = pd.DataFrame(img_list, columns=['Index', 'img_name', 'Class', 'img_path'])

step = 20000
fidx = 0
for curs in range(0, infodf.shape[0], step):
    cur_img = []
    curp = infodf[curs: curs + step]['img_path']
    for ipath in tqdm(curp, total=len(curp), desc='process image from {} to {}'.format(curs, curs + len(curp))):
        # image = cv2.imread(ipath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(open(ipath, 'rb'))
        image = image.convert('RGB')
        image = np.asarray(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        cur_img.append(image.transpose(2, 0, 1))
    savepath = os.path.join(save_path, trainp + str(fidx) + '.npy')
    curnp = np.stack(cur_img, axis=0)
    print(curnp.shape, curnp.dtype)
    np.save(savepath, curnp)
    fidx += 1

    del curnp, cur_img
    gc.collect()










