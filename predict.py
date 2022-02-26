from inspect import classify_class_attrs
import numpy as np
import json
import pandas as pd
from tqdm import trange
import argparse
from datasets.config import GeMConfig
from modules.gem import GeM
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from datasets.dl import GeMData
import torch
import os
from tqdm import tqdm, trange
import gc
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="ir", type=str,
                        help="Path of the output dir.")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--arch", default='resnet18', type=str)
parser.add_argument("--save_path", default='ir/para/model.ep0', type=str)
args = parser.parse_args()

modelp = os.path.join(args.save_path)
dbp = os.path.join(args.dpath, "database.npy")
testp = os.path.join(args.dpath, "test.npy")
imagep = os.path.join(args.dpath, "test_image.npy")
classp = os.path.join(args.dpath, "model_num.json")
resp = os.path.join(args.dpath, "mAP.json")
print('load data')
md = json.load(open(classp, "r"))
image_matrix = torch.ByteTensor(np.load(imagep))
testset = np.load(testp)
testinput = testset[:, 0]
test_class = testset[:, 1]
dataset = GeMData(image_matrix, torch.LongTensor(testinput))
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
print('load trained model')
model_info = GeMConfig()
model_info.arch = args.arch
model = GeM(model_info)
pretrained_model = torch.load(modelp, map_location='cpu')
print(model.load_state_dict(pretrained_model, strict=False))
model.to(0)

model.eval()  
batch_res = []
db_tensor = torch.FloatTensor(np.load(dbp).T).to(0)
with torch.no_grad():
    for data in tqdm(data_loader, total=len(data_loader), desc="generate vectors"):
        input_data = data / 255.0
        input_data = input_data.to(0)
        res = model.mips(input_data, 224, db_tensor)
        batch_res.append(res.cpu().numpy())

final_matrix = np.concatenate(batch_res, axis=0)
print(final_matrix.shape)

labeled = np.zeros(final_matrix.shape)
mAP_list = []
for i in tqdm(range(final_matrix.shape[0]), desc='map to binary and calculate mAP'):
    cur_s = set(md[test_class[i]])
    plist = []
    for j in range(final_matrix.shape[1]):
        labeled[i, j] = int(final_matrix[i][j] in cur_s)
        if labeled[i, j] == 1:
            plist.append(labeled[i, :j + 1].sum() / (j + 1))

    if len(plist) != 0:
        mAP_list.append(sum(plist) / len(plist))
    else:
        mAP_list.append(0)

print(sum(mAP_list) / len(mAP_list))
json.dump(mAP_list, open(resp, "w"))



