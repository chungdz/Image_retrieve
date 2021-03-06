import json
import pickle
import numpy as np
import os

class ModelConfig():
    def __init__(self):

        self.h = 224
        self.w = 224
        self.neg_count = 4
        self.hidden_size = 1000
        self.dropout = 0.5

class GeMConfig():
    def __init__(self, dpath):

        self.neg_count = 4
        self.progress = True
        self.md = json.load(open(os.path.join(dpath, 'model_num.json'), 'r'))
        self.class_num = len(self.md)

    def set_arch(self, arch):
        self.arch = arch
        if arch == 'resnet18':
            self.hidden_size = 512
        else:
            self.hidden_size = 2048



