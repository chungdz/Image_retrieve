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
        self.gem1 = MultiStageGeM(cfg.hidden_list[0], cfg.hidden_size)
        self.gem2 = MultiStageGeM(cfg.hidden_list[1], cfg.hidden_size)
        self.gem3 = MultiStageGeM(cfg.hidden_list[2], cfg.hidden_size)
        self.gem4 = MultiStageGeM(cfg.hidden_list[3], cfg.hidden_size)

        self.ln1 = nn.LayerNorm(cfg.hidden_list[0], eps=1e-6)
        self.ln2 = nn.LayerNorm(cfg.hidden_list[1], eps=1e-6)
        self.ln3 = nn.LayerNorm(cfg.hidden_list[2], eps=1e-6)
        self.ln4 = nn.LayerNorm(cfg.hidden_list[3], eps=1e-6)

        self.a1 = 0.1
        self.a2 = 0.2
        self.a3 = 0.4
        self.a4 = 0.8

    def forward(self, x):
        
        hidden_list = self.transformer(x)

        output1, thw1 = hidden_list[4]
        output2, thw2 = hidden_list[8]
        output3, thw3 = hidden_list[12]
        output4, thw4 = hidden_list[16]

        normed_output1 = self.ln1(output1)
        normed_output2 = self.ln2(output2)
        normed_output3 = self.ln3(output3)
        normed_output4 = self.ln4(output4)

        pooled1 = self.gem1(normed_output1.permute(0, 2, 1))
        pooled2 = self.gem2(normed_output2.permute(0, 2, 1))
        pooled3 = self.gem3(normed_output3.permute(0, 2, 1))
        pooled4 = self.gem4(normed_output4.permute(0, 2, 1))

        pooled = self.a1 * pooled1 + self.a2 * pooled2 + self.a3 * pooled3 + self.a4 * pooled4
        pooled_size = torch.linalg.vector_norm(pooled, ord=2, dim=-1, keepdim=True) + 1e-7
        return pooled / pooled_size
