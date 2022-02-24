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
from sklearn.metrics import roc_auc_score

def run(cfg, train_dataset, valid_dataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Build model.
    model = GeM(cfg.model_info)
    model.to(0)
    # for bug fixing
    # pretrained_model = torch.load('ir/para/model.ep0', map_location='cpu')
    # print(model.load_state_dict(pretrained_model, strict=False))
    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=cfg.lr_shrink)
    # Training and validation
    for epoch in range(cfg.epoch):
        print("lr in this epoch:", steplr.get_last_lr())
        # if epoch == 0:
        #     steplr.step()
        #     continue
        train(cfg, epoch, model, train_data_loader, optimizer, steps_one_epoch)
        validate(cfg, model, valid_data_loader)
        steplr.step()

def train(cfg, epoch, model, loader, optimizer, steps_one_epoch):
    model.train()
    model.zero_grad()
    enum_dataloader = tqdm(enumerate(loader), total=len(loader), desc="EP-{} train".format(epoch))
    mean_loss = 0
    input_label = torch.zeros((cfg.batch_size), dtype=torch.long).to(0)
    loss_list = []
    for index, data in enum_dataloader:
        # 1. Forward
        data = data / 255.0
        data = data.to(0)
        pred = model(data, 224)
        loss = F.cross_entropy(pred, input_label[:data.size(0)])

        # 3.Backward.
        loss.backward()
        # try gradient clipper to avoid gradient explosion
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        model.zero_grad()
        # index add
        mean_loss += loss.item()
        loss_list.append(loss.item())
        if str(loss.item()) == 'nan':
            print(loss_list[-100:])
            print(index)
            exit()
            
        if index % cfg.show_batch == 0 and index > 0:
            cur_mean_loss = mean_loss / cfg.show_batch
            enum_dataloader.set_description("EP-{} train, batch {} loss is {}".format(epoch, index, cur_mean_loss))
            mean_loss = 0
    
    torch.save(model.state_dict(), os.path.join(cfg.save_path, "model.ep{}".format(epoch)))

def validate(cfg, model, valid_data_loader):
    model.eval()  
        
    labels = []
    preds = []
    with torch.no_grad():
        for data in tqdm(valid_data_loader, total=len(valid_data_loader), desc="valid"):
            input_data = data[:, :-1] / 255.0
            label_data = data[:, -1]
            input_data = input_data.to(0)
            res = model(input_data, 224, valid_mode=True)
            labels += label_data.cpu().numpy().tolist()
            preds += res.cpu().numpy().tolist()
    print('running score')
    score = roc_auc_score(labels, preds)
    print(score)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--epoch", default=6, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--save_path", default='para', type=str)
parser.add_argument("--arch", default='resnet18', type=str)
parser.add_argument("--show_batch", default=1000, type=int)
parser.add_argument("--lr_shrink", default=0.5, type=float)
args = parser.parse_args()
print('load data')
matrixp = os.path.join(args.dpath, "imageset.npy")
trainsetp = os.path.join(args.dpath, "train.npy")
validsetp = os.path.join(args.dpath, "valid.npy")

args.model_info = GeMConfig()
args.model_info.arch = args.arch
pmatrix = torch.ByteTensor(np.load(matrixp))
trainset = torch.LongTensor(np.load(trainsetp))
validset = torch.LongTensor(np.load(validsetp))

train_dataset = GeMData(pmatrix, trainset)
valid_dataset = GeMData(pmatrix, validset, isValid=True)
run(args, train_dataset, valid_dataset)


