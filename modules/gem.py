import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision._internally_replaced_utils import load_state_dict_from_url

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

class GeM(nn.Module):

    def __init__(self, cfg):
        super(GeM, self).__init__()
        self.resnet = ResNetRaw(cfg.arch)
        # state_dict = load_state_dict_from_url(model_urls[cfg.arch], progress=cfg.progress)
        # print("Load {} pre-trained parameters from pytorch".format(cfg.arch), self.resnet.load_state_dict(state_dict))
        self.neg_count = cfg.neg_count
        self.hidden_size = cfg.hidden_size
        self.gem_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, len(cfg.cm))
        self.p = nn.Parameter(torch.Tensor([3]))
    
    def gem(self, x):
        xsize = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=False) + 1e-7
        xpower = torch.sum(torch.pow(x, self.p), dim=-1, keepdim=False)
        gem = torch.pow(xpower / xsize + 0.1, 1 / self.p)
        gem = self.gem_proj(gem)
        gem_size = torch.linalg.vector_norm(gem, ord=2, dim=-1, keepdim=True) + 1e-7
        return gem / gem_size
        # return gem
    
    def forward(self, data, l):
        batch_size = data.size(0)
        r = data.reshape(batch_size, 3, l, l)
        r = self.resnet(r)
        r = r.reshape(batch_size, self.hidden_size, -1)
        r = self.gem(r)
        outp = self.out_proj(r)

        return outp
    
    def predict(self, data, l):
        batch_size = data.size(0)
        r = data.reshape(batch_size, 3, l, l)
        r = self.resnet(r)
        r = r.reshape(batch_size, self.hidden_size, -1)
        r = self.gem(r)
        return r
    
    def mips(self, data, l, db, k=20):
        curq = self.predict(data, l)
        cur_score = torch.matmul(curq, db)
        _, topk = torch.topk(cur_score, k, dim=1)
        return topk

    

