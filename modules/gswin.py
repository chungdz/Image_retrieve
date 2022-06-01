import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin.build import build_model
from .swin.utils import load_pretrained
from .swin.config import get_config
from datasets.config import SwinConfig

class SpatialAttention(nn.Module):

    def __init__(self, insize, outsize):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding='same')
        self.proj = nn.Linear(insize, outsize)
    
    def forward(self, feature_maps):
        # input batch_size, channel, w, h
        meanp = feature_maps.mean(dim=1, keepdim=True)
        maxp, _ = feature_maps.max(dim=1, keepdim=True)
        cp = torch.cat([meanp, maxp], dim=1)
        att = torch.sigmoid(self.conv(cp))
        agg = torch.sum((feature_maps * att).reshape(feature_maps.size(0), feature_maps.size(1), -1), dim=-1)
        agg = self.proj(agg)
        return agg

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
        '''
        torch.Size([32, 3136, 192])
        torch.Size([32, 784, 384])
        torch.Size([32, 196, 768])
        torch.Size([32, 49, 1536])
        torch.Size([32, 49, 1536])
        '''
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
    
    def predict_all(self, x):
        '''
        torch.Size([32, 3136, 192])
        torch.Size([32, 784, 384])
        torch.Size([32, 196, 768])
        torch.Size([32, 49, 1536])
        torch.Size([32, 49, 1536])
        '''
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        
        to_return = []
        x = self.st.pos_drop(x)
        img = x.permute(0, 2, 1).reshape(x.size(0), x.size(1), 56, 56)
        to_return.append(img)
        x = self.st.layers[0](x)
        img = x.permute(0, 2, 1).reshape(x.size(0), x.size(1), 28, 28)
        to_return.append(img)
        x = self.st.layers[1](x)
        img = x.permute(0, 2, 1).reshape(x.size(0), x.size(1), 14, 14)
        to_return.append(img)
        x = self.st.layers[2](x)
        img = x.permute(0, 2, 1).reshape(x.size(0), x.size(1), 7, 7)
        to_return.append(img)
        x = self.st.layers[3](x)
        img = x.permute(0, 2, 1).reshape(x.size(0), x.size(1), 7, 7)
        to_return.append(img)

        return to_return


class SwinFMS(nn.Module):

    def __init__(self):
        super(SwinFMS, self).__init__()
        scfg = SwinConfig()
        cfg = get_config(scfg)
        self.st = build_model(cfg)
        load_pretrained(cfg, self.st)

        self.hidden = scfg.hidden
        self.gem = MultiStageGeM(self.hidden, self.hidden)

    def forward(self, x):
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        x = self.st.pos_drop(x)

        for layer in self.st.layers:
            x = layer(x)

        x = self.st.norm(x)  # B L C
        x = x.permute(0, 2, 1)
        
        pooled = self.gem(x)
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7

        return pooled / pooled_size


class SwinFMGL(nn.Module):

    def __init__(self):
        super(SwinFMGL, self).__init__()
        scfg = SwinConfig()
        cfg = get_config(scfg)
        self.st = build_model(cfg)
        load_pretrained(cfg, self.st)
        self.hidden = scfg.hidden
        self.scfg = scfg
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
        # self.conv1 = nn.Conv2d(scfg.channels[0], self.scfg.dc, self.scfg.dkernel[0], stride=self.scfg.dkernel[0], padding=0)
        # self.conv2 = nn.Conv2d(scfg.channels[1], self.scfg.dc, self.scfg.dkernel[1], stride=self.scfg.dkernel[1], padding=0)
        # self.conv3 = nn.Conv2d(scfg.channels[2], self.scfg.dc, self.scfg.dkernel[2], stride=self.scfg.dkernel[2], padding=0)
        # self.conv4 = nn.Conv2d(scfg.channels[3], self.scfg.dc, self.scfg.dkernel[3], stride=self.scfg.dkernel[3], padding=0)
        self.conv1 = SpatialAttention(scfg.channels[0], scfg.hidden)
        self.conv2 = SpatialAttention(scfg.channels[1], scfg.hidden)
        self.conv3 = SpatialAttention(scfg.channels[2], scfg.hidden)
        self.conv4 = SpatialAttention(scfg.channels[3], scfg.hidden)


    def forward(self, x):
        '''
        torch.Size([32, 3136, 192])
        torch.Size([32, 784, 384])
        torch.Size([32, 196, 768])
        torch.Size([32, 49, 1536])
        torch.Size([32, 49, 1536])
        '''
        x = self.st.patch_embed(x)
        if self.st.ape:
            x = x + self.st.absolute_pos_embed
        x = self.st.pos_drop(x)

        x = self.st.layers[0](x)
        to_add = self.ln1(x).permute(0, 2, 1)
        v1 = self.gem1(to_add)
        v1d = self.conv1(to_add.reshape(-1, self.scfg.channels[0], self.scfg.resolution[0], self.scfg.resolution[0]))
        v1 = torch.cat([v1, v1d.flatten(1)], dim=1)

        x = self.st.layers[1](x)
        to_add = self.ln2(x).permute(0, 2, 1)
        v2 = self.gem2(to_add)
        v2d = self.conv2(to_add.reshape(-1, self.scfg.channels[1], self.scfg.resolution[1], self.scfg.resolution[1]))
        v2 = torch.cat([v2, v2d.flatten(1)], dim=1)

        x = self.st.layers[2](x)
        to_add = self.ln3(x).permute(0, 2, 1)
        v3 = self.gem3(to_add)
        v3d = self.conv3(to_add.reshape(-1, self.scfg.channels[2], self.scfg.resolution[2], self.scfg.resolution[2]))
        v3 = torch.cat([v3, v3d.flatten(1)], dim=1)

        x = self.st.layers[3](x)
        to_add = self.ln4(x).permute(0, 2, 1)
        v4 = self.gem4(to_add)
        v4d = self.conv4(to_add.reshape(-1, self.scfg.channels[3], self.scfg.resolution[3], self.scfg.resolution[3]))
        v4 = torch.cat([v4, v4d.flatten(1)], dim=1)

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

