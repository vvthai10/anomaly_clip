import torch.nn as nn


class SegAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(SegAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        y = self.fc1(x)
        return y


class DetAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(DetAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        y = self.fc1(x)
        return y


class AnomalyCLIP_VisionLearner(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.seg_adapters = nn.ModuleList(
            [SegAdapter(1024, bottleneck=768) for i in range(len(features))]
        )

        self.DetAdapter = nn.ModuleList(
            [SegAdapter(1024, bottleneck=768) for i in range(len(features))]
        )

    def encoder_vision(self, image_features, patch_features):
        det_feats = []
        seg_feats = []
        for idx, patch_feature in enumerate(patch_features):
            seg_feat = self.seg_adapters[idx].forward(patch_feature)
            seg_feats.append(seg_feat.permute(1, 0, 2))

            det_feat = self.seg_adapters[idx].forward(patch_feature)
            det_feats.append(det_feat.permute(1, 0, 2))

        return det_feats, seg_feats
