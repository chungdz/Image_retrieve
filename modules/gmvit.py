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

class MViTMulti(nn.Module):

    def __init__(self, cfg) -> None:
        super(MViTMulti, self).__init__()
        # load backbone
        model_info = get_cfg()
        model_info.merge_from_file(cfg.model_settings_path)
        self.transformer = MViT(model_info)
        pretrained_model = torch.load(cfg.model_pretrained_path, map_location='cpu')
        print('load MViT trained parameters', self.transformer.load_state_dict(pretrained_model['model_state'], strict=False))
        # init other modules
        gem_list = []
        ln_list = []
        for i in range(17):
            gem_list.append(MultiStageGeM(cfg.hidden_list[i], cfg.hidden_size))
            ln_list.append(nn.LayerNorm(cfg.hidden_list[i], eps=1e-6))
        self.gems = nn.ModuleList(gem_list)
        self.lns = nn.ModuleList(ln_list)

    def forward(self, x):
        
        hidden_list = self.transformer(x)

        to_add1, thw1 = hidden_list[0]
        to_add1 = self.lns[0](to_add1)
        to_add1 = to_add1.permute(0, 2, 1)
        to_add1 = self.gems[0](to_add1[:, :, 1:])
        for i in range(1, 17):
            cur_add, cur_thw = hidden_list[i]
            cur_add = self.lns[i](cur_add)
            cur_add = cur_add.permute(0, 2, 1)
            cur_add = self.gems[i](cur_add[:, :, 1:])
            to_add1 = to_add1 * 0.9 + cur_add

        pooled = to_add1
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size
