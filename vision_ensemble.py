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


class AnomalyCLIP_VisionLearner(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        # self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

    def encode_image_learn(self, x):
        # image = self.image_encoder.conv1(image)
        #         # image = image.reshape(image.shape[0], image.shape[1], -1)
        #         # image = image.permute(0, 2, 1)
        #         #
        #         # image = torch.cat(
        #         #     [self.image_encoder.class_embedding.to(image.dtype) + torch.zeros(image.shape[0], 1, image.shape[-1],
        #         #                                                                       dtype=image.dtype,
        #         #                                                                       device=image.device),
        #         #      image], dim=1)
        #         # image = image + self.image_encoder.positional_embedding.to(image.dtype)
        #         #
        #         # image = self.image_encoder.patch_dropout(image)
        #         # image = self.image_encoder.ln_pre(image)
        #         #
        #         # image = image.permute(1, 0, 2)

        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        #####################################################################################
        side = int((self.image_encoder.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.image_encoder.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.image_encoder.positional_embedding.data = torch.cat([self.image_encoder.positional_embedding[:1, :], new_pos[0]], 0)
        #####################################################################################

        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_out = []
        seg_patch_tokens = []
        # det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                x = self.image_encoder.transformer.resblocks[i](x)
                # attn_out.append(attn)
            else:
                x = self.image_encoder.transformer.resblocks[i](x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                # det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.9 * x + 0.1 * seg_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                # det_patch_tokens.append(det_adapt_med)

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
        # det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        x = x.permute(1, 0, 2)
        # pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(x[:, 0, :])

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens
