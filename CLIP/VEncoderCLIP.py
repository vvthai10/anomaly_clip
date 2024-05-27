import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from collections import OrderedDict


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


class VEncoderCLIP(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

    def encode_image_learn(self, image):
        image = self.image_encoder.conv1(image)
        image = image.reshape(image.shape[0], image.shape[1], -1)
        image = image.permute(0, 2, 1)

        image = torch.cat(
            [self.image_encoder.class_embedding.to(image.dtype) + torch.zeros(image.shape[0], 1, image.shape[-1],
                                                                              dtype=image.dtype,
                                                                              device=image.device),
             image], dim=1)
        image = image + self.image_encoder.positional_embedding.to(image.dtype)

        image = self.image_encoder.patch_dropout(image)
        image = self.image_encoder.ln_pre(image)

        image = image.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                image, attn = self.image_encoder.transformer.resblocks[i](image, attn_mask=None)
                attn_out.append(attn)
            else:
                image, attn_map = self.image_encoder.transformer.resblocks[i](image, attn_mask=None)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](image)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](image)

                image = 0.8 * image + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        # TODO: Handle attention_map
        # B, C, L = attn_out[0].shape
        # H = int(math.sqrt(L - 1))
        #
        # # out_attn = torch.zeros([H, H]).to('cuda')
        # out_attn = []
        # for i in range(len(attn_out)):
        #     batch_out_attn = torch.zeros([H, H]).to('cuda')
        #     for b in range(B):
        #         batch_out_attn = batch_out_attn + attn_out[i][b, 0, 1:].view(H, H)
        #     out_attn.append(batch_out_attn.unsqueeze(0))
        # out_attn = torch.cat(out_attn)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        image = image.permute(1, 0, 2)
        pooled, tokens = self.image_encoder._global_pool(image)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens
