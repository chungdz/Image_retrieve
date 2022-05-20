import numpy as np
from PIL import Image
import pickle
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse

class DisjointS:
    def __init__(self) -> None:
        self.nodes = {}
        self.rank = 0

    def make_set(self, v):
        assert(v not in self.nodes)
        self.nodes[v] = v
        self.rank += 1
    
    def isExist(self, v):
        return v in self.nodes
    
    def find_set(self, v):
        assert(v in self.nodes)
        cp = self.nodes[v]
        if cp == v:
            return v
        
        self.nodes[v] = self.find_set(cp)
        return self.nodes[v]
    
    def union(self, x, y):
        nx = self.find_set(x)
        ny = self.find_set(y)

        if nx == ny:
            return False
        else:
            self.nodes[ny] = nx
            self.rank -= 1
        
        return True

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="landmark", type=str,
                        help="root path of all data")
parser.add_argument("--img_size", default=224, type=int,
                        help="image size")
args = parser.parse_args()

ox_path = os.path.join('revisitop', 'gnd_roxford5k.pkl')
pa_path = os.path.join('revisitop', 'gnd_rparis6k.pkl')
ox_folder = os.path.join(args.dpath, 'oxford5k')
pa_folder = os.path.join(args.dpath, 'paris6k')
tmp = os.path.join(args.dpath, 'trainv3.npy')
tip = os.path.join(args.dpath, 'train_info_v3.csv')

ox_dict = pickle.load(open(ox_path, 'rb'))
pa_dict = pickle.load(open(pa_path, 'rb'))
ds = DisjointS()

for info in ox_dict['gnd']:
    cur_list = info['easy'] + info['hard']
    father_name = ox_dict['imlist'][cur_list[0]]
    father_path = os.path.join(ox_folder, father_name + '.jpg')
    for idx in cur_list:
        img_name = ox_dict['imlist'][idx]
        img_path = os.path.join(ox_folder, img_name + '.jpg')
        if not ds.isExist(img_path):
            ds.make_set(img_path)
        ds.union(father_path, img_path)

for info in pa_dict['gnd']:
    cur_list = info['easy'] + info['hard']
    father_name = pa_dict['imlist'][cur_list[0]]
    father_path = os.path.join(pa_folder, father_name + '.jpg')
    for idx in cur_list:
        img_name = pa_dict['imlist'][idx]
        img_path = os.path.join(pa_folder, img_name + '.jpg')
        if not ds.isExist(img_path):
            ds.make_set(img_path)
        ds.union(father_path, img_path)

fdict = {}
class_idx = 0
for v in ds.nodes:
    curf = ds.find_set(v)
    if curf not in fdict:
        fdict[curf] = class_idx
        class_idx += 1

class_dict = {}
for v in ds.nodes:
    class_dict[v] = fdict[ds.find_set(v)]

train_matrix = []
train_info = []
cur_idx = 0
for img_path, iclass in tqdm(class_dict.items(), total=len(class_dict)):
    img = Image.open(open(img_path, 'rb')).convert('RGB')
    iarray = np.asarray(img)
    niarray = cv2.resize(iarray, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    niarray = niarray.transpose(2, 0, 1)
    train_matrix.append(niarray)
    train_info.append([img_path, iclass, cur_idx])
    cur_idx += 1

tm = np.stack(train_matrix, axis=0)
infodf = pd.DataFrame(train_info, columns=['img_path', 'Class', 'Index'])

print(tm.shape, infodf.shape)

np.save(tmp, tm)
infodf.to_csv(tip, index=None)
