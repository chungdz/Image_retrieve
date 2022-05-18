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

        self.gem1 = MultiStageGeM(self.hidden, self.hidden)
        self.gem2 = MultiStageGeM(self.hidden, self.hidden)
        self.gem3 = MultiStageGeM(self.hidden, self.hidden)
        self.gem4 = MultiStageGeM(self.hidden, self.hidden)

        self.a1 = 0.1
        self.a2 = 0.2
        self.a3 = 0.4
        self.a4 = 0.8
    
    def forward(self, x):
        
        deit_output = self.transformer(x, output_hidden_states=True)
        all_hidden = deit_output.hidden_states

        hidden1 = all_hidden[3].permute(0, 2, 1)
        hidden2 = all_hidden[6].permute(0, 2, 1)
        hidden3 = all_hidden[9].permute(0, 2, 1)
        hidden4 = all_hidden[12].permute(0, 2, 1)

        pooled1 = self.gem1(hidden1)
        pooled2 = self.gem2(hidden2)
        pooled3 = self.gem3(hidden3)
        pooled4 = self.gem4(hidden4)

        pooled = self.a1 * pooled1 + self.a2 * pooled2 + self.a3 * pooled3 + self.a4 * pooled4
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size