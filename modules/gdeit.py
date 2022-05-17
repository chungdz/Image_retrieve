import torch
import torch.nn as nn
from transformers import DeiTModel, DeiTConfig

class DeiTRaw(nn.Module):

    def __init__(self):
        super(DeiTRaw, self).__init__()
        self.transformer = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
    
    def forward(self, x):
        
        x = self.transformer(x)
        pooler_output = x.pooler_output
        size = torch.linalg.vector_norm(pooler_output, ord=2, dim=-1, keepdim=True) + 1e-7

        return pooler_output / size