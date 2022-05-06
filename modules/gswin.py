import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin.build import build_model
from .swin.utils import load_pretrained
from .swin.config import get_config
from datasets.config import SwinConfig

class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding='same')
    
    def forward(self, feature_maps):
        # input batch_size, channel, w, h
        meanp = feature_maps.mean(dim=1, keepdim=True)
        maxp, _ = feature_maps.max(dim=1, keepdim=True)
        cp = torch.cat([meanp, maxp], dim=1)
        att = torch.sigmoid(self.conv(cp))
        return att

class MultiStageGeM(nn.Module):

    def __init__(self, insize, outsize) -> None:
        super(MultiStageGeM, self).__init__()
        self.proj = nn.Linear(insize, outsize)
        self.p = nn.Parameter(torch.Tensor([3]))
        self.minimumx = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)
    
    def forward(self, x):
        # x should be B C H*W
        # C should be equal to insize
        xpower = torch.pow(torch.maximum(x, self.minimumx), self.p)
        gem = torch.pow(xpower.mean(dim=-1, keepdim=False), 1.0 / self.p)
        gem = self.proj(gem)
        return torch.tanh(gem)

class SwinFM(nn.Module):

    def __init__(self):
        super(SwinFM, self).__init__()
        scfg = SwinConfig()
        cfg = get_config(scfg)
        self.st = build_model(cfg)
        load_pretrained(cfg, self.st)
        
        self.gem1 = MultiStageGeM(scfg.channels[0], scfg.hidden)
        self.gem2 = MultiStageGeM(scfg.channels[1], scfg.hidden)
        self.gem3 = MultiStageGeM(scfg.channels[2], scfg.hidden)
        self.gem4 = MultiStageGeM(scfg.channels[3], scfg.hidden)
    
    def forward(self, x):
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        x = self.st.pos_drop(x)

        x = self.st.layers[0](x)
        cur_v = self.gem1(x.permute(0, 2, 1))

        x = self.st.layers[1](x)
        cur_v += self.gem2(x.permute(0, 2, 1))

        x = self.st.layers[2](x)
        cur_v += self.gem3(x.permute(0, 2, 1))

        x = self.st.layers[3](x)
        cur_v += self.gem4(x.permute(0, 2, 1))

        gem_size = torch.linalg.vector_norm(cur_v, ord=2, dim=-1, keepdim=True) + 1e-7

        # x = self.st.norm(x)  # B L C
        # x = x.permute(0, 2, 1)
        # x = x.reshape(-1, x.size(1), 7, 7)
        # x = self.bn(x)
        return cur_v / gem_size

    

