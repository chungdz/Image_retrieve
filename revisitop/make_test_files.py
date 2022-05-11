from PIL import Image
import argparse
import os
from tqdm import tqdm
import pickle
import json
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="landmark", type=str,
                        help="root path of all data")
parser.add_argument("--dname", default="oxford5k", type=str,
                        help="dataset type, the other is paris6k")
parser.add_argument("--img_size", default=224, type=int,
                        help="image size")
args = parser.parse_args()

img_folder = os.path.join(args.dpath, args.dname)
dict_path = os.path.join('revisitop', 'gnd_r{}.pkl'.format(args.dname))
qm_path = os.path.join(args.dpath, '{}_qm.npy'.format(args.dname))
dbm_path = os.path.join(args.dpath, '{}_dbm.npy'.format(args.dname))
info_path = os.path.join(args.dpath, '{}_info.json'.format(args.dname))

cfg = pickle.load(open(dict_path, 'rb'))

query_list = []
for img_name in tqdm(cfg['qimlist'], desc='process query images and resize'):
    img_path = os.path.join(img_folder, img_name + '.jpg')
    img = Image.open(open(img_path, 'rb')).convert('RGB')
    iarray = np.asarray(img)
    niarray = cv2.resize(iarray, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    niarray = niarray.transpose(2, 0, 1)
    query_list.append(niarray)
qm = np.stack(query_list, axis=0)

db_list = []
for img_name in tqdm(cfg['imlist'], desc='process database images and resize'):
    img_path = os.path.join(img_folder, img_name + '.jpg')
    img = Image.open(open(img_path, 'rb')).convert('RGB')
    iarray = np.asarray(img)
    niarray = cv2.resize(iarray, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    niarray = niarray.transpose(2, 0, 1)
    db_list.append(niarray)
dbm = np.stack(db_list, axis=0)

print(qm.shape, dbm.shape)

json.dump(cfg['gnd'], open(info_path, 'w', encoding='utf8'))
np.save(qm_path, qm)
np.save(dbm_path, dbm)


