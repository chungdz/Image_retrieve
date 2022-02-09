from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
 
class FNNData(Dataset):
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.pic_matrix = cfg.pic_matrix
        self.dataset = np.load(dataset)

    def __getitem__(self, index):
        return torch.FloatTensor(self.pic_matrix[self.dataset[index]])
 
    def __len__(self):
        return self.dataset.shape[0]

