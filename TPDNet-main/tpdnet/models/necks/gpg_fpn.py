"""
GPG_3 Neck for CLRNet
Complete implementation from SkinLesionSegmentation_V2
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ..registry import NECKS


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    """Generate DCT filters"""
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    """MultiSpectral Attention Layer from FcaNet"""
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiFrequencyChannelAttention(nn.Module):
    """Multi-Frequency Channel Attention"""
    def __init__(self, in_channels, dct_h, dct_w, frequency_branches=16, frequency_selection='top', reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        assert len(mapper_x) == len(mapper_y)
        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), 
                                self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x_pooled = x
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.named_buffers():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                # Remove .half() to avoid dtype issues, keep original dtype
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)
        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)
        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class SeparableConv2d(nn.Module):
    """Separable Convolution"""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


@NECKS.register_module
class GPG_FPN(nn.Module):
    """
    GPG_3 Neck for CLRNet
    Complete implementation with multi-scale context enhancement and frequency domain attention
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=3,
                 att_channel=None,
                 att_h=None,
                 att_w=None,
                 frequency_selection='top',
                 frequency_branches=16,
                 width=512,
                 up_kwargs=None,
                 norm_layer=nn.BatchNorm2d,
                 cfg=None):
        super(GPG_FPN, self).__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) >= 3, "GPG_FPN requires at least 3 input feature levels"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.width = width
        
        # Use last 3 levels (P3, P4, P5)
        self.gpg_in_channels = in_channels[-3:]
        
        # Calculate attention dimensions if not provided
        # For CULane: img_h=320, img_w=800, so P5 is about 10x25
        if att_h is None or att_w is None:
            # Estimate from typical feature map sizes
            # P5: stride=32, so 320/32=10, 800/32=25
            att_h = 10
            att_w = 25
        if att_channel is None:
            att_channel = width
        
        if up_kwargs is None:
            up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.up_kwargs = up_kwargs
        
        # GPG_3 module for the last 3 levels
        self.gpg_3 = GPG_3_Module(
            in_channels=self.gpg_in_channels,
            att_channel=att_channel,
            att_h=att_h,
            att_w=att_w,
            frequency_selection=frequency_selection,
            frequency_branches=frequency_branches,
            width=width,
            up_kwargs=up_kwargs,
            norm_layer=norm_layer
        )
        
        # Output projection to match out_channels
        # GPG outputs one feature (width channels), we create num_outs outputs from it
        self.output_convs = nn.ModuleList()
        for i in range(num_outs):
            if i == 0:
                # First output: direct GPG output projection
                self.output_convs.append(nn.Sequential(
                    nn.Conv2d(width, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                # Additional outputs: from upsampled previous output
                self.output_convs.append(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)
        
        # If inputs have more levels than in_channels, take the last len(in_channels) levels
        # This matches FPN behavior: only use the levels specified in in_channels
        if len(inputs) > len(self.in_channels):
            inputs = inputs[-len(self.in_channels):]
        
        # GPG_3 processes the last 3 levels (P3, P4, P5)
        # Since in_channels=[128, 256, 512] has 3 levels, inputs should now have 3 levels
        assert len(inputs) == 3, f"Expected 3 input levels after filtering, got {len(inputs)}"
        
        gpg_inputs = inputs
        gpg_output = self.gpg_3(*gpg_inputs)
        
        # Build outputs: GPG outputs one feature, we create num_outs outputs from it
        outs = []
        
        # First output: direct GPG output
        gpg_out = self.output_convs[0](gpg_output)
        outs.append(gpg_out)
        
        # Create additional outputs by upsampling
        for i in range(1, self.num_outs):
            # Upsample from previous output
            upsampled = F.interpolate(outs[-1], scale_factor=2, **self.up_kwargs)
            outs.append(self.output_convs[i](upsampled))
        
        return tuple(outs[:self.num_outs])


class GPG_3_Module(nn.Module):
    """GPG_3 Module - Multi-scale context enhancement with frequency domain attention"""
    def __init__(self, in_channels, att_channel, att_h, att_w, frequency_selection='top', 
                 frequency_branches=16, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3_Module, self).__init__()
        self.up_kwargs = up_kwargs if up_kwargs else {'mode': 'bilinear', 'align_corners': True}
        
        # Input projections
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        
        # Multi-dilation convolutions
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        
        # Gate convolution
        self._gate_conv2 = nn.Sequential(
            nn.Conv2d(width, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Output convolutions
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, dilation=1, kernel_size=3, padding=1)
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(width * 4, width, dilation=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_2 = nn.ModuleList([
            nn.Conv2d(width // 2, 3, dilation=1, kernel_size=3, padding=1),
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False)
        ])
        
        # Frequency domain attention
        self.att = MultiSpectralAttentionLayer(att_channel, att_h, att_w, reduction=16, freq_sel_method='top16')
        self.att_channel = MultiFrequencyChannelAttention(att_channel, att_h, att_w, frequency_branches, frequency_selection)
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(att_channel, att_channel // 16, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(att_channel // 16, att_channel, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, *inputs):
        # Project inputs to same width
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, c, h, w = feats[-1].size()
        # Resize to same spatial size
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        
        # Concatenate and apply multi-dilation convolutions
        feat = torch.cat(feats, dim=1)
        feat1 = self.dilation1(feat)
        feat2 = self.dilation2(feat)
        feat3 = self.dilation3(feat)
        feat0 = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Compute adaptive weights
        weight = self.conv_out(feat0)
        weight = self.conv3x3_2[0](weight)
        weight = F.softmax(weight, dim=1)
        
        weight_1 = weight[:, 0, :, :].unsqueeze(1)
        weight_2 = weight[:, 1, :, :].unsqueeze(1)
        weight_3 = weight[:, 2, :, :].unsqueeze(1)
        
        # Apply frequency domain attention to each dilation branch
        weight_att1 = self.att(feat1)
        avg1 = self.avg_pool(weight_att1)
        mx1 = self.max_pool(weight_att1)
        avg1 = self.fc1(avg1)
        mx1 = self.fc1(mx1)
        avg1 = self.relu(avg1)
        mx1 = self.relu(mx1)
        avg1 = self.fc2(avg1)
        mx1 = self.fc2(mx1)
        weight_feat1 = self.sigmoid_channel(avg1 + mx1)
        att_feat1 = feat1 * weight_feat1 + feat1
        
        weight_att2 = self.att(feat2)
        avg2 = self.avg_pool(weight_att2)
        mx2 = self.max_pool(weight_att2)
        avg2 = self.fc1(avg2)
        mx2 = self.fc1(mx2)
        avg2 = self.relu(avg2)
        mx2 = self.relu(mx2)
        avg2 = self.fc2(avg2)
        mx2 = self.fc2(mx2)
        weight_feat2 = self.sigmoid_channel(avg2 + mx2)
        att_feat2 = feat2 * weight_feat2 + feat2
        
        weight_att3 = self.att(feat3)
        avg3 = self.avg_pool(weight_att3)
        mx3 = self.max_pool(weight_att3)
        avg3 = self.fc1(avg3)
        mx3 = self.fc1(mx3)
        avg3 = self.relu(avg3)
        mx3 = self.relu(mx3)
        avg3 = self.fc2(avg3)
        mx3 = self.fc2(mx3)
        weight_feat3 = self.sigmoid_channel(avg3 + mx3)
        att_feat3 = feat3 * weight_feat3 + feat3
        
        # Adaptive fusion
        fusion = (weight_1 * feat1) + (weight_2 * feat2) + (weight_3 * feat3)
        fusion2 = self._gate_conv2(fusion)
        input_features = (fusion2 * feat0) + feat0
        input_features = self.conv3x3_2[1](input_features)
        
        # Final concatenation and output
        input_features = torch.cat([input_features, att_feat1, att_feat2, att_feat3], dim=1)
        input_features = self.conv_out1(input_features)
        return input_features

