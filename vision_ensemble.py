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
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.seg_adapters = nn.ModuleList([SegAdapter(1024, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([SegAdapter(1024, bottleneck=768) for i in range(len(features))])

    def encoder_vision(self, det_patch_features, seg_patch_features):

        update_patch_seg_features = []
        for idx, patch_feature in enumerate(seg_patch_features):
            update_patch_seg_feature = self.seg_adapters[idx].forward(patch_feature)
            update_patch_seg_features.append(update_patch_seg_feature.permute(1, 0, 2))

        update_patch_det_features = []
        for idx, patch_feature in enumerate(det_patch_features):
            update_patch_det_feature = self.det_adapters[idx].forward(patch_feature)
            update_patch_det_features.append(update_patch_det_feature.permute(1, 0, 2))

        return update_patch_det_features, update_patch_seg_features
