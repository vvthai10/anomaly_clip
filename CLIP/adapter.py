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


# Residual CLIP Adapter
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


class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features, prompt_parameters):
        super().__init__()
        self.clipmodel = clip_model
        self.clip_state_dict = clip_model.state_dict()
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )

        # TODO: Handle init Transformer module
        self.embed_dim = self.clip_state_dict["text_projection"].shape[1]
        self.context_length = self.clip_state_dict["positional_embedding"].shape[0]
        self.transformer_width = self.clip_state_dict["ln_final.weight"].shape[0]
        self.transformer_layers = self.transformer_width // 64
        self.transformer_heads = len(set(k.split(".")[2] for k in self.clip_state_dict if k.startswith(f"transformer.resblocks")))
        self.transformer = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )
        self.vocab_size = self.clip_state_dict["token_embedding.weight"].shape[0]
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.ln_final = LayerNorm(self.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def encode_image_learn(self, image):
        image = self.image_encoder.conv1(image)
        image = image.reshape(image.shape[0], image.shape[1], -1)
        image = image.permute(0, 2, 1)

        image = torch.cat(
            [self.image_encoder.class_embedding.to(image.dtype) + torch.zeros(image.shape[0], 1, image.shape[-1], dtype=image.dtype,
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

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L - 1))

        # out_attn = torch.zeros([H, H]).to('cuda')
        out_attn = []
        for i in range(len(attn_out)):
            batch_out_attn = torch.zeros([H, H]).to('cuda')
            for b in range(B):
                batch_out_attn = batch_out_attn + attn_out[i][b, 0, 1:].view(H, H)
            out_attn.append(batch_out_attn.unsqueeze(0))
        out_attn = torch.cat(out_attn)
        image = image.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(image)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text=None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        # x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        # x = x + self.positional_embedding.to(cast_dtype)

        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("test", x.shape, len(deep_compound_prompts_text))
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L-1))
        out_attn = torch.zeros([H, H]).to('cuda')

        # out_attn = torch.zeros([H, H]).to('cuda')
        out_attn = []
        for i in range(len(attn_out)):
            layer_out_attn = []
            for b in range(B):
                layer_out_attn.append(attn_out[i][b, 0, 1:].view(H, H).unsqueeze(0))
            out_attn.append(torch.cat(layer_out_attn))
            # out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens




