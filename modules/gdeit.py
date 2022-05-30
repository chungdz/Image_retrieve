import torch
import torch.nn as nn
from transformers import DeiTModel, DeiTConfig
from .gswin import MultiStageGeM

class DeiTRaw(nn.Module):

    def __init__(self):
        super(DeiTRaw, self).__init__()
        self.transformer = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
    
    def forward(self, x):
        
        x = self.transformer(x)
        pooler_output = x.pooler_output
        size = torch.linalg.vector_norm(pooler_output, ord=2, dim=-1, keepdim=True) + 1e-7

        return pooler_output / size

class DeiTGeM(nn.Module):

    def __init__(self, cfg):
        super(DeiTGeM, self).__init__()
        self.transformer = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.hidden = cfg.hidden_size
        self.gem = MultiStageGeM(self.hidden, self.hidden)
    
    def forward(self, x):
        
        deit_output = self.transformer(x, output_hidden_states=True)
        all_hidden = deit_output.hidden_states
        last_hidden = all_hidden[-1]
        x = last_hidden.permute(0, 2, 1)
        pooled = self.gem(x)
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size

class DeiTMultiGeM(nn.Module):

    def __init__(self, cfg):
        super(DeiTMultiGeM, self).__init__()
        self.transformer = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.hidden = cfg.hidden_size

        self.gems = nn.ModuleList([MultiStageGeM(self.hidden, self.hidden) for _ in range(13)])
    
    def forward(self, x):
        
        deit_output = self.transformer(x, output_hidden_states=True)
        all_hidden = deit_output.hidden_states

        to_add = self.gems[0](all_hidden[0].permute(0, 2, 1))
        for i in range(1, 13):
            to_add = to_add * 0.5 + self.gems[i](all_hidden[i].permute(0, 2, 1))

        pooled = to_add
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size