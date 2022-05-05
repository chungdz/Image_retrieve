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

class SwinFM(nn.Module):

    def __init__(self):
        super(SwinFM, self).__init__()
        cfg = get_config(SwinConfig())
        self.st = build_model(cfg)
        load_pretrained(cfg, self.st)
    
    def forward(self, x):
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        x = self.st.pos_drop(x)

        for layer in self.st.layers:
            x = layer(x)

        x = self.st.norm(x)  # B L C
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.size(1), 7, 7)
        return x

    

