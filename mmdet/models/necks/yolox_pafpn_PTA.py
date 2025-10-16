# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..layers import CSPLayer


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class PTA_Layers(nn.Module):
    '''
    Partial Temporal Aggregation Layer
    -----------------------------------designed by ChongWang
    '''
    def __init__(self,in_channels, out_channels, T, partial,left_partial,group):
        super().__init__()
        self.T = T
        self.partial=partial
        self.left_partial=left_partial
        ##为了式子简洁，起个短别名
        p = partial
        lp= left_partial
        rp = partial-left_partial

        self.pta_layers = []
        for in_c,out_c in zip(in_channels, out_channels):
            assert in_c%partial==0 and out_c%partial==0
            if lp !=0:
                left_layer=nn.Sequential(
                    nn.Conv2d(in_channels=in_c*lp//p, out_channels=out_c*lp//p,kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(out_c*lp//p),
                    nn.SiLU(inplace=True)
                )
            else:
                left_layer=IdentityLayer()

            if rp !=0:
                right_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_c*rp//p, out_channels=out_c*rp//p,kernel_size=(3,3,3),
                              stride=1,padding=1, groups=group,bias=True),
                    nn.BatchNorm3d(out_c*rp//p),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(in_channels=out_c*rp//p, out_channels=out_c*rp//p, kernel_size=(1, 3, 3),
                              stride=1, padding=(0,1,1), groups=1, bias=True),
                    nn.BatchNorm3d(out_c*rp//p),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(in_channels=out_c * rp // p, out_channels=out_c * rp // p, kernel_size=(3, 1, 1),
                              stride=1, padding=(1, 0, 0), groups=1, bias=True),
                    nn.BatchNorm3d(out_c * rp // p),
                    nn.SiLU(inplace=True)
                )
            else:
                right_layer = IdentityLayer()

            final_conv=nn.Sequential(
                nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1,stride=1,padding=0,bias=True),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True)
            )
            self.pta_layers.append(nn.ModuleList([left_layer,right_layer,final_conv]))
        self.pta_layers=nn.ModuleList(self.pta_layers)

    def forward(self,feats):
        out_all = []
        for i in range(len(feats)):
            BT, C, H, W = feats[i].size()
            feat = feats[i].view(BT//self.T,self.T, C,H,W).contiguous()#B,T,C,H,W
            feat = feat.permute(0,2,1,3,4).contiguous()#B,C,T,H,W

            feat_l, feat_r = torch.split(feat,
                                         split_size_or_sections=[self.left_partial*C//self.partial,
                                                                 (self.partial-self.left_partial)*C//self.partial],
                                         dim=1)
            feat_l = feat_l[:,:,(self.T-1)//2,...]
            out_l = self.pta_layers[i][0](feat_l)

            out_r = self.pta_layers[i][1](feat_r)
            out_r = torch.max(out_r,dim=2,keepdim=False)[0]

            out = self.pta_layers[i][2](torch.cat([out_l,out_r],dim=1))
            out_all.append(out)
        return out_all


@MODELS.register_module()
class YOLOXPAFPN_PTA(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ptas_cfg,
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
        super(YOLOXPAFPN_PTA, self).__init__(init_cfg)

        self.pta_modules = PTA_Layers(**ptas_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
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

        inputs = self.pta_modules(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
