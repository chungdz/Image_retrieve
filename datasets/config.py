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
        elif arch == 'resnet50':
            self.hidden_size = 2048
        elif arch == 'resnet101':
            self.hidden_size = 2048
        elif arch == 'swin':
            self.hidden_size = 1536

class SwinConfig:
    def __init__(self):
        self.cfg = './swin_para/swin_large_patch4_window7_224_22k.yaml'
        self.changeto1k = 'swin_para/map22kto1k.txt'
        self.opts = None
        self.batch_size = 32
        self.data_path = None
        self.zip = True
        self.cache_mode = 'part'
        self.pretrained = './swin_para/swin_large_patch4_window7_224_22k.pth'
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = True
        self.amp_opt_level = 'O1'
        self.output = 'output'
        self.tag = None
        self.eval = True
        self.throughput = True
        self.local_rank=0



