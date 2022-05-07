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
        return gem

class SwinFM(nn.Module):

    def __init__(self):
        super(SwinFM, self).__init__()
        scfg = SwinConfig()
        cfg = get_config(scfg)
        self.st = build_model(cfg)
        load_pretrained(cfg, self.st)
        self.hidden = scfg.hidden
        # self.a1 = nn.Parameter(torch.Tensor([0.25]))
        # self.a2 = nn.Parameter(torch.Tensor([0.25]))
        # self.a3 = nn.Parameter(torch.Tensor([0.25]))
        # self.a4 = nn.Parameter(torch.Tensor([0.25]))
        self.a1 = 0.1
        self.a2 = 0.2
        self.a3 = 0.4
        self.a4 = 0.8
        self.ln1 = nn.LayerNorm(scfg.channels[0])
        self.ln2 = nn.LayerNorm(scfg.channels[1])
        self.ln3 = nn.LayerNorm(scfg.channels[2])
        self.ln4 = nn.LayerNorm(scfg.channels[3])
        
        self.gem1 = MultiStageGeM(scfg.channels[0], scfg.hidden)
        self.gem2 = MultiStageGeM(scfg.channels[1], scfg.hidden)
        self.gem3 = MultiStageGeM(scfg.channels[2], scfg.hidden)
        self.gem4 = MultiStageGeM(scfg.channels[3], scfg.hidden)
        # self.proj = nn.Sequential(
        #     nn.Linear(scfg.hidden * 4, scfg.hidden * 2),
        #     nn.Tanh(),
        #     nn.Linear(scfg.hidden * 2, 4),
        #     nn.Sigmoid()
        # )
        
    
    def forward(self, x):
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        x = self.st.pos_drop(x)

        x = self.st.layers[0](x)
        to_add = self.ln1(x)
        v1 = self.gem1(to_add.permute(0, 2, 1))

        x = self.st.layers[1](x)
        to_add = self.ln2(x)
        v2 = self.gem2(to_add.permute(0, 2, 1))

        x = self.st.layers[2](x)
        to_add = self.ln3(x)
        v3 = self.gem3(to_add.permute(0, 2, 1))

        x = self.st.layers[3](x)
        to_add = self.ln4(x)
        v4 = self.gem4(to_add.permute(0, 2, 1))

        # final_cat = torch.cat([v1, v2, v3, v4], dim=-1)
        # final_score = self.proj(final_cat)
        # final_cat = final_cat.reshape(-1, 4, self.hidden)
        # final = final_cat * final_score.unsqueeze(-1)
        # final = final.sum(dim=1, keepdim=False)

        final = self.a1 * v1 + self.a2 * v2 + self.a3 * v3 + self.a4 * v4
        gem_size = torch.linalg.vector_norm(final, ord=2, dim=-1, keepdim=True) + 1e-7

        # x = self.st.norm(x)  # B L C
        # x = x.permute(0, 2, 1)
        # x = x.reshape(-1, x.size(1), 7, 7)
        # x = self.bn(x)
        return final / gem_size

    

