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
from tqdm import tqdm
import gc
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="ir", type=str, help="Path of the output dir.")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--arch", default='resnet18', type=str)
parser.add_argument("--save_path", default='ir/para/model.ep0', type=str)
parser.add_argument("--input", default="imageset.npy", type=str, help="input file")
parser.add_argument("--output", default="database.npy", type=str, help="output file")
args = parser.parse_args()

matrixp = os.path.join(args.dpath, args.input)
modelp = os.path.join(args.save_path)
dbp = os.path.join(args.dpath, args.output)
print('load data')
pmatrix = torch.ByteTensor(np.load(matrixp))
indexlist = torch.arange(pmatrix.size(0))
dataset = GeMData(pmatrix, indexlist)
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
with torch.no_grad():
    for data in tqdm(data_loader, total=len(data_loader), desc="generate vectors"):
        input_data = data / 255.0
        input_data = input_data.to(0)
        res = model.predict(input_data, 224)
        batch_res.append(res.cpu().numpy())

final_matrix = np.concatenate(batch_res, axis=0)
print(final_matrix.shape)
np.save(dbp, final_matrix)




