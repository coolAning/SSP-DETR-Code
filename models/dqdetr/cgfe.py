# Modified from https://github.com/Jongchan/attention-module
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(
        self,
        bn: bool = True,
        bias: bool = False,
        gn: bool = False,
        use_density: bool = False,
    ):
        super(SpatialGate, self).__init__()
        if bn and gn:
            raise ValueError("SpatialGate: bn and gn cannot both be enabled")
        kernel_size = 7
        self.use_density = bool(use_density)
        self.gn = bool(gn)
        self.compress = ChannelPool()
        self.spatial = Conv_BN(
            2,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
            bn=bn,
            bias=bias,
        )
        self.spatial_gn = nn.GroupNorm(1, 1, affine=True) if self.gn else None
        self.spatial_den = None
        self.spatial_den_gn = None
        if self.use_density:
            self.spatial_den = Conv_BN(
                1,
                1,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                relu=False,
                bn=bn,
                bias=bias,
            )
            self.spatial_den_gn = nn.GroupNorm(1, 1, affine=True) if self.gn else None

    def forward(self, x, density_logit=None, alpha_eff: float = 0.0):
        x_compress = self.compress(x)
        s_cbam = self.spatial(x_compress)
        if self.spatial_gn is not None:
            s_cbam = self.spatial_gn(s_cbam)
        s = s_cbam
        if density_logit is not None and self.spatial_den is not None:
            den = density_logit
            if den.dim() == 3:
                den = den.unsqueeze(1)
            den = den.to(dtype=s_cbam.dtype)
            s_den = self.spatial_den(den)
            if self.spatial_den_gn is not None:
                s_den = self.spatial_den_gn(s_den)
            if isinstance(alpha_eff, torch.Tensor):
                a = alpha_eff.to(device=s_cbam.device, dtype=s_cbam.dtype)
            else:
                a = torch.as_tensor(float(alpha_eff), device=s_cbam.device, dtype=s_cbam.dtype)
            s = s_cbam + a * s_den
        scale = F.sigmoid(s) # broadcasting
        return scale


class CGFE(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=['avg', 'max'],
        no_spatial=False,
        no_channel=False,
        num_feature_levels=4,
        spatial_gate_bn: bool = True,
        spatial_gate_bias: bool = False,
        spatial_gate_gn: bool = False,
        spatial_density_enable: bool = False,
    ):
        super(CGFE, self).__init__()
        self.num_feat = num_feature_levels
        self.ChannelGate = None
        if not no_channel:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(
                bn=spatial_gate_bn,
                bias=spatial_gate_bias,
                gn=spatial_gate_gn,
                use_density=bool(spatial_density_enable),
            )
            
    def forward(self, x, memory, spatial_shapes, density_logits_per_level=None, alpha_eff: float = 0.0):
        feats = []
        idx = 0
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape

        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs, 256, h, w)
            if not self.no_spatial:
                dlogit = None
                if density_logits_per_level is not None and len(density_logits_per_level) > i:
                    dlogit = density_logits_per_level[i]
                c2 = self.SpatialGate(x[i], density_logit=dlogit, alpha_eff=alpha_eff)
                feat = feat * c2
            if self.ChannelGate is not None:
                c1 = self.ChannelGate(feat)
                feat = feat * c1
            feat = feat.flatten(2).transpose(1, 2)
            feats.append(feat)
            idx += h*w
            
        x_out = torch.cat(feats, 1)
        return x_out
        
        
class MultiScaleFeature(nn.Module):
    def __init__(self, channels=256, is_5_scale=False):
        super(MultiScaleFeature, self).__init__()
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale
        
    def forward(self, x):
        x_out = []
        x_out.append(x)
        x = self.conv1(x)
        x_out.append(x)
        x = self.conv2(x)
        x_out.append(x)
        x = self.conv3(x)
        x_out.append(x)
        
        if self.is_5_scale:
           x = self.conv4(x)
           x_out.append(x) 
        return x_out
