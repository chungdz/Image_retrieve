import torch
import torch.nn as nn

class DeiTRaw(nn.Module):

    def __init__(self):
        super(DeiTRaw, self).__init__()
        # resnet 50
        self.transformer = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.transformer(x)

        return x