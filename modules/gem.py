import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision._internally_replaced_utils import load_state_dict_from_url
from .gswin import SwinFM

class ResNetRaw(ResNet):

    def __init__(self, arch):
        if arch == 'resnet18':
            b = BasicBlock
            layers = [2, 2, 2, 2]
        else:
            b = Bottleneck
            layers = [3, 4, 6, 3]
        super(ResNetRaw, self).__init__(b, layers, num_classes=1000)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

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

class GeM(nn.Module):

    def __init__(self, cfg):
        super(GeM, self).__init__()
        if 'resnet' in cfg.arch:
            self.backbone = ResNetRaw(cfg.arch)
            state_dict = load_state_dict_from_url(model_urls[cfg.arch], progress=cfg.progress)
            print("Load {} pre-trained parameters from pytorch".format(cfg.arch), self.backbone.load_state_dict(state_dict))
        else:
            self.backbone = SwinFM()
        self.neg_count = cfg.neg_count
        self.hidden_size = cfg.hidden_size
        self.gem_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.p = nn.Parameter(torch.Tensor([3]))
        self.minimumx = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)
        self.sa = SpatialAttention()
        self.fc1 = nn.Linear(self.hidden_size, cfg.class_num)
    
    # def gem(self, x):
    #     xsize = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=False) + 1e-7
    #     xpower = torch.sum(torch.pow(x, self.p), dim=-1, keepdim=False)
    #     gem = torch.pow(xpower / xsize + 0.1, 1 / self.p)
    #     gem = self.gem_proj(gem)
    #     gem_size = torch.linalg.vector_norm(gem, ord=2, dim=-1, keepdim=True) + 1e-7
    #     return gem / gem_size
    # new GEM from DELG
    def gem(self, x):
        xpower = torch.pow(torch.maximum(x, self.minimumx), self.p)
        gem = torch.pow(xpower.mean(dim=-1, keepdim=False), 1.0 / self.p)
        gem = self.gem_proj(gem)
        gem_size = torch.linalg.vector_norm(gem, ord=2, dim=-1, keepdim=True) + 1e-7
        return gem / gem_size
    
    def gem_no_norm(self, x):
        xsize = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=False) + 1e-7
        xpower = torch.sum(torch.pow(x, self.p), dim=-1, keepdim=False)
        gem = torch.pow(xpower / xsize + 0.1, 1 / self.p)
        return gem
    
    def forward(self, data, l, valid_mode=False, scale=1):
        '''
        data for train is like
        torch.Size([32, 6, 3, 224, 224])
        r1 after resnet is like
        torch.Size([32, 512, 7, 7])
        r1 after gem is like
        torch.Size([32, 512])
        r1 after repeating is like
        torch.Size([32, 5, 512])

        data for valid is like
        torch.Size([32, 301056])
        torch.Size([32, 2, 3, 224, 224])
        '''
        if valid_mode:
            negc = 0
        else:
            negc = self.neg_count
        
        batch_size = data.size(0)
        data = data.reshape(batch_size, 2 + negc, 3, l, l)
        if scale != 1:
            newl = int(round(scale * l))
            data = data.reshape(batch_size * (2 + negc), 3, l, l)
            ndata = F.interpolate(data, newl, mode='bilinear', align_corners=True)
            ndata = ndata.reshape(batch_size, 2 + negc, 3, newl, newl)
            l = newl
            data = ndata

        r1 = data[:, 0].reshape(batch_size, 3, l, l)
        r1 = self.backbone(r1)
        r1 = r1.reshape(batch_size, self.hidden_size, -1)
        r1 = self.gem(r1)
        r1 = r1.repeat(1, negc + 1).view(batch_size, negc + 1, self.hidden_size)

        r2 = data[:, 1:].reshape(batch_size * (negc + 1), 3, l, l)
        r2 = self.backbone(r2)
        r2 = r2.reshape(batch_size, negc + 1, self.hidden_size, -1)
        r2 = self.gem(r2)
        r2 = r2.reshape(batch_size, negc + 1, self.hidden_size)

        return torch.sum(r1 * r2, dim=-1)
    
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


    

