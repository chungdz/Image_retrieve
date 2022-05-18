import torch
import torch.nn as nn
import torch.nn.functional as F
from .mvit.mvit import MViT
from .mvit.defaults import get_cfg
from .gswin import MultiStageGeM

class MViTRaw(nn.Module):

    def __init__(self, cfg) -> None:
        super(MViTRaw, self).__init__()
        # load backbone
        model_info = get_cfg()
        model_info.merge_from_file(cfg.model_settings_path)
        self.transformer = MViT(model_info)
        pretrained_model = torch.load(cfg.model_pretrained_path, map_location='cpu')
        print('load MViT trained parameters', self.transformer.load_state_dict(pretrained_model['model_state'], strict=False))
        # init other modules
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        
        hidden_list = self.transformer(x)
        last_output, last_thw = hidden_list[-1]
        normed_output = self.transformer.norm(last_output)
        tcls = normed_output[:, 0]

        pooler_output = self.dense(tcls)
        pooler_output = self.activation(pooler_output)
        size = torch.linalg.vector_norm(pooler_output, ord=2, dim=-1, keepdim=True) + 1e-7

        return pooler_output / size

class MViTGeM(nn.Module):

    def __init__(self, cfg) -> None:
        super(MViTGeM, self).__init__()
        # load backbone
        model_info = get_cfg()
        model_info.merge_from_file(cfg.model_settings_path)
        self.transformer = MViT(model_info)
        pretrained_model = torch.load(cfg.model_pretrained_path, map_location='cpu')
        print('load MViT trained parameters', self.transformer.load_state_dict(pretrained_model['model_state'], strict=False))
        # init other modules
        self.gem = MultiStageGeM(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        
        hidden_list = self.transformer(x)
        last_output, last_thw = hidden_list[-1]
        normed_output = self.transformer.norm(last_output)
        to_pool = normed_output.permute(0, 2, 1)        

        pooled = self.gem(to_pool)
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size