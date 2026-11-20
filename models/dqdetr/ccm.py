import torch
import torch.nn as nn


class CCMFeatureExtractor(nn.Module):
    """
    CCM feature extractor.

    Takes encoder `memory` and returns a level-0 spatial feature map [B, 256, H0, W0]
    used by density / CGFE modules. Any scalar counting prediction is removed.
    """
    def __init__(self):
        super().__init__()
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        self.in_channels = 512
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)

    def forward(self, features, spatial_shapes=None):
        # features: memory [B, sum(hw), C]
        features = features.transpose(1, 2)
        bs, _, _ = features.shape
        if spatial_shapes is None:
            raise ValueError("spatial_shapes must be provided for CCMFeatureExtractor")
        h = int(spatial_shapes[0][0].item())
        w = int(spatial_shapes[0][1].item())
        hw = h * w

        v_feat = features[:, :, 0:hw].view(bs, 256, h, w)
        x = self.conv1(v_feat)
        x = self.ccm(x)
        return x


class SpatialPriorHead(nn.Module):
    """Predict unnormalized logits for the spatial object prior P(x)."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, ccm_feature: torch.Tensor):
        feat = self.shared(ccm_feature)
        return self.p_head(feat)


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
