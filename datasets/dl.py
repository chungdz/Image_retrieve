from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
 
class FNNData(Dataset):
    def __init__(self, cfg, isValidation=False):
        self.cfg = cfg
        self.pic_matrix = torch.LongTensor(cfg.pic_matrix)
        self.dataset = torch.LongTensor(cfg.dataset)
        self.isValidation = isValidation

    def __getitem__(self, index):
        if self.isValidation:
            img = self.pic_matrix[self.dataset[index][0]].reshape(-1)
            targets = self.dataset[index][1:]
            return torch.cat([img, targets], dim=-1)
        return torch.LongTensor(self.pic_matrix[self.dataset[index]])
 
    def __len__(self):
        return self.dataset.shape[0]

class GeMData(Dataset):
    def __init__(self, pm, ds, isValid=False):
        '''
        change in matrix to matrix path when really do it
        '''
        self.pic_matrix = pm
        self.dataset = ds
        self.isValid = isValid

    def __getitem__(self, index):
        if self.isValid:
            img = self.pic_matrix[self.dataset[index][:2]].reshape(-1)
            label = self.dataset[index][2].reshape(-1)
            return torch.cat([img, label], dim=-1)
        
        return self.pic_matrix[self.dataset[index]]
 
    def __len__(self):
        return self.dataset.shape[0]


