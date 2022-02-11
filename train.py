import numpy as np
import json
import pandas as pd
from tqdm import trange
import argparse
from modules.classical_multi import MultiResNet
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from datasets.dl import FNNData
import torch
import os
from tqdm import tqdm
import gc
import torch.nn.functional as F

def run(cfg, train_dataset, valid_dataset, fp):
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
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Build model.
    model = FNNData(cfg)
    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    min_ebay = -1
    # Training and validation
    
    for epoch in range(cfg.epoch):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, model, train_data_loader, optimizer, steps_one_epoch)
        eval_return, rc, preds = validate(cfg, model, valid_data_loader)
        print(epoch, eval_return)
        
        if min_ebay == -1 or eval_return < min_ebay:
            min_ebay = eval_return
            min_rc = rc
            min_pred = preds
            torch.save(model.state_dict(), fp)
        
    return min_ebay, min_rc, min_pred


def train(cfg, epoch, model, loader, optimizer, steps_one_epoch):
    model.train()
    model.zero_grad()
    enum_dataloader = tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch))
    index = 0
    mean_loss = 0
    for data in enum_dataloader:
        if index >= steps_one_epoch:
            break
        
        # 1. Forward
        pred = model(data[:, :-2]).squeeze()
        loss = F.cross_entropy(pred, data[:, -2].long())

        # 3.Backward.
        loss.backward()

        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        model.zero_grad()
        # index add
        index += 1
        mean_loss += loss
        if index % cfg.show_batch == 0 and index > 0:
            cur_mean_loss = mean_loss / cfg.show_batch
            enum_dataloader.set_description("EP-{} train, batch {} loss is {}".format(epoch, index, cur_mean_loss))
            mean_loss = 0
            torch.save(model.state_dict(), 'para/fnn_tmp')

def validate(cfg, model, valid_data_loader):
    model.eval()  
        
    with torch.no_grad():
        preds, truths, rks = list(), list(), list()
        for data in valid_data_loader:
            # 1. Forward
            pred = model(data[:, :-2])
            pred = pred / (torch.sum(pred, dim=1, keepdim=True))
            pred = pred * torch.arange(pred.size(1)).unsqueeze(0)
            pred = torch.sum(pred, dim=1)
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.numpy().tolist()
            except:
                preds.append(int(pred.cpu().numpy()))
            truths += data[:, -2].numpy().flatten().tolist()
            rks += data[:, -1].numpy().flatten().tolist()

        rpreds = np.round(preds)
        residual = (truths - rpreds).astype("int")
        loss = np.where(residual < 0, residual * -0.6, residual * 0.4)

        return np.mean(loss), rks, preds

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=10, type=int)
parser.add_argument("--epoch", default=1, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=int)
parser.add_argument("--save_path", default='para', type=str)
parser.add_argument("--show_batch", default=1000, type=int)
args = parser.parse_args()
cate_info = json.load(open('data/category_info.json'))
args.cate_info = cate_info

loss_and_output = []
total_rc = []
total_preds = []
for i in range(1, args.folds + 1):
    print('model:', i)
    train_dataset = FNNData('data/subtrain_cat/train_{}.tsv'.format(i))
    valid_dataset = FNNData('data/subtrain_cat/valid_{}.tsv'.format(i))
    cur_loss, rc, preds = run(args, train_dataset, valid_dataset, os.path.join(args.save_path, 'pfnn_{}'.format(i)))
    loss_and_output.append(cur_loss)
    total_rc += rc
    total_preds += preds
    del train_dataset, valid_dataset
    gc.collect()

to_save = []
for rnumber, predict_value in zip(total_rc, total_preds):
    to_save.append([rnumber, predict_value])
savedf = pd.DataFrame(to_save, columns=['record_number', 'pFNN_predict'])
savedf.to_csv('data/sl_data/pfnn_train.tsv', sep='\t', index=None) 

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/pfnn_weight.json', 'w'))
json.dump(loss_and_output, open('para/all_pfnn_log.json', 'w'))
print('mean loss:', np.mean(loss_and_output))

