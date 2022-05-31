import torch
import torch.nn as nn
import torch.nn.functional as F
from .gswin import SwinFM, SwinFMS, SwinFMGL
from .gresnet import ResNetRaw, ResNetRawS
from .gdeit import DeiTRaw, DeiTGeM, DeiTMultiGeM
from .gmvit import MViTRaw, MViTGeM, MViTMulti

class GeM(nn.Module):

    def __init__(self, cfg):
        super(GeM, self).__init__()
        if 'resnet' in cfg.arch:
            if cfg.isM:
                print('load multi stage GeM ResNet')
                self.backbone = ResNetRaw(cfg.arch)
            else:
                print('load single stage GeM ResNet')
                self.backbone = ResNetRawS(cfg.arch)
        elif 'deit' == cfg.arch:
            print('load deit')
            self.backbone = DeiTRaw()
        elif 'deitgem' == cfg.arch:
            print('load deit gem')
            self.backbone = DeiTGeM(cfg)
        elif 'deitmulti' == cfg.arch:
            print('load deit multi gem')
            self.backbone = DeiTMultiGeM(cfg)
        elif 'swin' == cfg.arch:
            if cfg.isM:
                print('load multi stage GeM Swin Transformer')
                self.backbone = SwinFM()
            else:
                print('load single stage GeM Swin Transformer')
                self.backbone = SwinFMS()
        elif 'swingl' == cfg.arch:
            print('load single stage GeM Swin Transformer with local')
            self.backbone = SwinFMGL()
        elif 'mvit' == cfg.arch:
            print('load MViT CLS')
            self.backbone = MViTRaw(cfg)
        elif 'mvitgem' == cfg.arch:
            print('load MViT Single GeM')
            self.backbone = MViTGeM(cfg)
        elif 'mvitmulti' == cfg.arch:
            print('load MViT Multi GeM')
            self.backbone = MViTMulti(cfg)
        
        self.neg_count = cfg.neg_count
        self.hidden_size = cfg.hidden_size
        self.fc1 = nn.Linear(self.hidden_size, cfg.class_num)
    
    def predict(self, data, l):
        batch_size = data.size(0)
        data = data.reshape(batch_size, 3, l, l)
        r = self.backbone(data)
        return r

    def mips(self, data, l, db, k=20):
        curq = self.predict(data, l)
        cur_score = torch.matmul(curq, db)
        _, topk = torch.topk(cur_score, k, dim=1)
        return topk
    
    def predict_class(self, data, l, scale=1):

        batch_size = data.size(0)
        data = data.reshape(batch_size, 3, l, l)
        if scale != 1:
            newl = int(round(scale * l))
            ndata = F.interpolate(data, newl, mode='bilinear', align_corners=True)
            l = newl
            data = ndata

        r1 = self.backbone(data)
        sscore = self.fc1(r1)
        return sscore


    

