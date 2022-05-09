import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls, resnet50, resnet101
from torchvision._internally_replaced_utils import load_state_dict_from_url
from .gswin import MultiStageGeM

class ResNetRaw(nn.Module):

    def __init__(self, arch):
        super(ResNetRaw, self).__init__()
        # resnet 50
        if arch == 'resnet50':
            print('load 50')
            self.resnet = resnet50(pretrained=True)
        elif arch == 'resnet101':
            print('load 101')
            self.resnet = resnet101(pretrained=True)

        self.a1 = 0.1
        self.a2 = 0.2
        self.a3 = 0.4
        self.a4 = 0.8
        self.sizelist = [256, 512, 1024, 2048]
        self.hidden = 2048

        # self.ln1 = nn.LayerNorm(self.sizelist[0])
        # self.ln2 = nn.LayerNorm(self.sizelist[1])
        # self.ln3 = nn.LayerNorm(self.sizelist[2])
        # self.ln4 = nn.LayerNorm(self.sizelist[3])
        
        self.gem1 = MultiStageGeM(self.sizelist[0], self.hidden)
        self.gem2 = MultiStageGeM(self.sizelist[1], self.hidden)
        self.gem3 = MultiStageGeM(self.sizelist[2], self.hidden)
        self.gem4 = MultiStageGeM(self.sizelist[3], self.hidden)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # torch.Size([32, 256, 32, 32])
        # torch.Size([32, 512, 16, 16])
        # torch.Size([32, 1024, 8, 8])
        # torch.Size([32, 2048, 4, 4])
        batch_size = x.size(0)

        x = self.resnet.layer1(x)
        to_add = x.reshape(batch_size, self.sizelist[0], -1)
        v1 = self.gem1(to_add)

        x = self.resnet.layer2(x)
        to_add = x.reshape(batch_size, self.sizelist[1], -1)
        v2 = self.gem2(to_add)

        x = self.resnet.layer3(x)
        to_add = x.reshape(batch_size, self.sizelist[2], -1)
        v3 = self.gem3(to_add)

        x = self.resnet.layer4(x)
        to_add = x.reshape(batch_size, self.sizelist[3], -1)
        v4 = self.gem4(to_add)

        final = self.a1 * v1 + self.a2 * v2 + self.a3 * v3 + self.a4 * v4
        gem_size = torch.linalg.vector_norm(final, ord=2, dim=-1, keepdim=True) + 1e-7

        return final / gem_size


class ResNetRawS(nn.Module):

    def __init__(self, arch):
        super(ResNetRawS, self).__init__()
        # resnet 50
        if arch == 'resnet50':
            print('load 50')
            self.resnet = resnet50(pretrained=True)
        elif arch == 'resnet101':
            print('load 101')
            self.resnet = resnet101(pretrained=True)
        self.hidden = 2048
        self.gem = MultiStageGeM(self.hidden, self.hidden)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        batch_size = x.size(0)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        pooled = x.reshape(batch_size, self.hidden, -1)
        pooled = self.gem(pooled)
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7

        return pooled / pooled_size
