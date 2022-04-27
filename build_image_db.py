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
parser.add_argument("--dpath", default="ir", type=str, help="root path of all data")
parser.add_argument("--batch_size", default=32, type=int, help="encoding batch size used in Pytorch DataLoader")
parser.add_argument("--multi_scale", default=0, type=int, help="whether to use multi-scale")
parser.add_argument("--arch", default='resnet18', type=str, help="backbone of model, should be same the training model")
parser.add_argument("--encoder", default='gem', type=str, help="encoder type gem or att")
parser.add_argument("--save_path", default='ir/para/model.ep0', type=str, help="where to load model parameters")
parser.add_argument("--input", default="imageset.npy", type=str, help="image matrix")
parser.add_argument("--output", default="database.npy", type=str, help="encoded image vectors")
parser.add_argument("--img_size", default=224, type=int, help="size of img")
args = parser.parse_args()

matrixp = os.path.join(args.dpath, args.input)
modelp = os.path.join(args.save_path)
dbp = os.path.join(args.dpath, args.output)
print('load data')
pmatrix = torch.ByteTensor(np.load(matrixp))
indexlist = torch.arange(pmatrix.size(0))
dataset = GeMData(pmatrix, indexlist)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
print('load trained model')
model_info = GeMConfig(args.dpath)
model_info.set_arch(args.arch)
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
        if args.multi_scale == 1:
            res = model.predict(input_data, args.img_size, scale_list=[1.0, 1.4147], encoder=args.encoder)
        else:
            res = model.predict(input_data, args.img_size, encoder=args.encoder)
        batch_res.append(res.cpu().numpy())

final_matrix = np.concatenate(batch_res, axis=0)
print(final_matrix.shape)
np.save(dbp, final_matrix)




