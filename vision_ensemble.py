import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn


class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class SegAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(SegAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 768, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        y = self.fc1(x)
        return y

class DetAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(DetAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(768, 384, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(384, 768, bias=False),
            nn.SiLU(inplace=False)
        )

    def forward(self, x):
        y = self.fc1(x)
        return y

class AnomalyCLIP_VisionLearner(nn.Module):
    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.visual = model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.visual.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.visual.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.visual.positional_embedding.data = torch.cat([self.visual.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.visual.positional_embedding.to(x.dtype)
        x = x + pos
        x = self.visual.ln_pre(x)

        # TODO: Adapter +
        x = x.permute(1, 0, 2)
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            with torch.no_grad():
                x = self.visual.transformer.resblocks[i](x)
            if (i + 1) in self.features:
                x_vv, x_ori = x
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x_vv)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x_ori)

                x_vv = 0.9 * x_vv + 0.1 * seg_adapt_out
                x_ori = 0.9 * x_ori + 0.1 * det_adapt_out
                x = [x_vv, x_ori]

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        _, x_ori = x
        x_ori = x_ori.permute(1, 0, 2)
        pooled = self.visual.ln_post(x_ori[:, 0, :])

        format_seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        format_det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        return pooled, format_det_patch_tokens, format_seg_patch_tokens