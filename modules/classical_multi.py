import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import resnet18

class MultiVgg(nn.Module):
    def __init__(self, cfg):
        super(MultiVgg, self).__init__()
        # self.vgg = VGG(make_layers(cfgs["A"], batch_norm=False), num_classes=cfg.hidden_size, )
        self.vgg = vgg11(pretrained=False, progress=True, num_classes=cfg.hidden_size)
        self.w = cfg.w
        self.h = cfg.h
        self.neg_count = cfg.neg_count
        self.hidden_size = cfg.hidden_size
        
    def forward(self, data, test_mode=False):
        if test_mode:
            return self.vgg(data)

        r1 = data[:, 0].reshape(-1, 3, self.h, self.w)
        r1 = self.vgg(r1)
        r1 = r1.repeat(1, self.neg_count + 1).view(-1, self.neg_count + 1, self.hidden_size)

        r2 = data[:, 1:].reshape(-1, 3, self.h, self.w)
        r2 = self.vgg(r2)
        r2 = r2.reshape(-1, self.neg_count + 1, self.hidden_size)

        return torch.sum(r1 * r2, dim=-1)

class MultiResNet(nn.Module):
    def __init__(self, cfg):
        super(MultiResNet, self).__init__()
        self.vgg = resnet18(pretrained=False, progress=True, num_classes=cfg.hidden_size)
        self.w = cfg.w
        self.h = cfg.h
        self.neg_count = cfg.neg_count
        self.hidden_size = cfg.hidden_size
        
    def forward(self, data, test_mode=False):
        if test_mode:
            return self.vgg(data)

        r1 = data[:, 0].reshape(-1, 3, self.h, self.w)
        r1 = self.vgg(r1)
        r1 = r1.repeat(1, self.neg_count + 1).view(-1, self.neg_count + 1, self.hidden_size)

        r2 = data[:, 1:].reshape(-1, 3, self.h, self.w)
        r2 = self.vgg(r2)
        r2 = r2.reshape(-1, self.neg_count + 1, self.hidden_size)

        return torch.sum(r1 * r2, dim=-1)

