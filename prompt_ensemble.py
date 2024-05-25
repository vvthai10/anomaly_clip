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

import math

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                # Here cls is the number of classes, n_ctx_pos represents the length of the learnable token,
                # and ctx_dim represents the dimension of the prompt.
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)



    def forward(self, cls_id =None):
        
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        return prompts, tokenized_prompts, self.compound_prompts_text


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


class AnomalyCLIP_VisionLearner(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        # self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

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
        # det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                image, attn = self.image_encoder.transformer.resblocks[i](image, attn_mask=None)
                attn_out.append(attn)
            else:
                image, attn_map = self.image_encoder.transformer.resblocks[i](image, attn_mask=None)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](image)
                # det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](image)

                image = 0.9 * image + 0.1 * seg_adapt_out # + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                # det_patch_tokens.append(det_adapt_med)

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
        # det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(image)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens #, det_patch_tokens

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
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
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L - 1))
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