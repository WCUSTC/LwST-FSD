# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..layers import CSPLayer
from ..detectors.yolox_VID import Pseudo3DDCN

class MyWrapper(nn.Module):
    def __init__(self, sequential):
        super(MyWrapper, self).__init__()
        self.sequential = sequential

    def forward(self, x1, x2):
        x = self.sequential[0](x1, x2)
        for layer in self.sequential[1:]:
            x = layer(x)
        return x

@MODELS.register_module()
class YOLOXPAFPN_DCN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 offset_cfg,
                 pdcn_cfg,
                 T,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN_DCN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = T
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        """# build top-down blocks"""
        self.upsample = nn.Upsample(**upsample_cfg)
        ##conv的输入减半，另外一半用于求offsets
        self.reduce_layers=nn.ModuleList([
            ConvModule(in_channels[2]//2, in_channels[1], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),#
            ConvModule(in_channels[1] // 2, in_channels[0], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels[0], in_channels[0], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            ])
        ## 最底层的那个CSPLayer输出改为256（而不是原先的128），这样CSPLayer后面的特征就能分为两半。
        self.top_down_blocks = nn.ModuleList([
            CSPLayer(in_channels[1] * 2, in_channels[1], num_blocks=num_csp_blocks, add_identity=False,
                     use_depthwise=use_depthwise, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            CSPLayer(in_channels[0] * 2, in_channels[0] * 2, num_blocks=num_csp_blocks, add_identity=False,
                     use_depthwise=use_depthwise, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
        ])

        ##三个offset layers
        self.offset_layers = nn.ModuleList()
        for level_cfg in offset_cfg:
            offset_layers_onelevel = []
            for layer in level_cfg:
                func = getattr(torch.nn, layer["type"])
                del layer["type"]
                offset_layers_onelevel.append(func(**layer))
            self.offset_layers.append(torch.nn.Sequential(*offset_layers_onelevel))

        ##三个 PDCN layers
        self.pdcn_layers = nn.ModuleList()
        for level_cfg in pdcn_cfg:
            pdcn_layers_onelevel = []
            for layer in level_cfg:
                if "torch.nn" in layer["type"]:
                    func = getattr(torch.nn, layer["type"].replace("torch.nn.",""))
                else:
                    func = globals()[layer["type"]]
                del layer["type"]
                pdcn_layers_onelevel.append(func(**layer))
            self.pdcn_layers.append(MyWrapper(torch.nn.Sequential(*pdcn_layers_onelevel)))

        """# build bottom-up blocks"""
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        """# top-down path"""
        offsets = []
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            BT,C,H,W = inner_outs[0].size()
            feat_heigh_1,feat_heigh_2 = inner_outs[0][:,:C//2,...], inner_outs[0][:,C//2:,...]
            feat_low = inputs[idx - 1]

            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh_1)
            inner_outs[0] = feat_heigh

            ##offset:[B, K, T, H, W]
            feat_heigh_2 = feat_heigh_2.view(BT//self.T, self.T, -1, H, W).contiguous()
            feat_heigh_2 = feat_heigh_2.permute(0,2,1,3,4).contiguous()
            offsets.append(self.offset_layers[len(self.in_channels) - 1 - idx](feat_heigh_2))

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        ##还有最高层特征的split---offset
        ##                  ---conv
        BT, C, H, W = inner_outs[0].size()
        feat_heigh_1, feat_heigh_2 = inner_outs[0][:, :C // 2, ...], inner_outs[0][:, C // 2:, ...]
        feat_heigh = self.reduce_layers[2](feat_heigh_1)
        inner_outs[0] = feat_heigh
        ##offset:[B, K, T, H, W]
        feat_heigh_2 = feat_heigh_2.view(BT // self.T, self.T, -1, H, W).contiguous()
        feat_heigh_2 = feat_heigh_2.permute(0, 2, 1, 3, 4).contiguous()
        offsets.append(self.offset_layers[2](feat_heigh_2))
        offsets = offsets[::-1]

        """在这里执行P-DCN
            伪的3D-DCN, offset只能再W,H 两个维度上
            输入 feat 与 offset
            feat: [B, C, T, H, W]
            offset:[B, K, T, H, W]
        """
        inner_outs_new=[]
        offsets_new = []
        for i,(offset, feat) in enumerate(zip(offsets,inner_outs)):
            # BT, K, H, W = offset.size()
            # offset = offset.view(-1, self.T, K, H, W).contiguous()
            # offset = torch.permute(offset,[0,2,1,3,4]).contiguous()
            offsets_new.append(offset)

            BT, C, H, W = feat.size()
            feat = feat.view(-1, self.T, C, H, W).contiguous()
            feat = torch.permute(feat,[0,2,1,3,4]).contiguous()#[B, C, T, H, W]
            feat = feat + self.pdcn_layers[len(self.in_channels)-1-i](feat,offset)
            inner_outs_new.append(feat[:,:,(self.T - 1) // 2, :, :])  # 在这里坍塌时间序列

        """# bottom-up path"""
        outs = [inner_outs_new[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs_new[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs),offsets_new
