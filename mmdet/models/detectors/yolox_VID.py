# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import sys,copy
sys.path.append(r"E:\mmdetection320\mmdet\models\layers\D3Dnet-master\code")
from dcn.modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from dcn.modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d
C3DCNs=['DeformConv_d', '_DeformConv']  #只允许使用自由度更高的 xxxx_d, 不允许内部自带的offset。
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch,os,random,cv2
import mmcv.ops.deform_conv
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from ..utils import samplelist_boxtype2tensor

sys.path.append(r"E:\mmdetection320\tools_VID\xfeat_modules")
from xfeat import XFeat

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

import torch.distributed as dist
# xfeat_inst = XFeat()
# if dist.is_initialized():
#     print("进程组已经初始化")
#     if dist.get_rank() == 0:
#         # 只在主节点（rank为0）上加载权重并广播给其他节点
#         for p in xfeat_inst.parameters():
#             dist.broadcast(p.data, src=0)
#     else:
#         # 非主节点等待接收权重
#         for p in xfeat_inst.parameters():
#             dist.broadcast(p.data, src=0)
# else:
#     pass
# xfeat_inst.eval()


@MODELS.register_module()
class YOLOXTSM_PTA(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        B,T,C,H,W = batch_inputs.size()
        batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
        x_lst = self.backbone(batch_inputs)
        if self.with_neck:
            x_lst = self.neck(x_lst)
        return x_lst


@MODELS.register_module()
class YOLOXTSM_Center(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        B,T,C,H,W = batch_inputs.size()
        batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
        x_lst = self.backbone(batch_inputs)
        x_lst_new = []
        for i, x in enumerate(x_lst):
            BT, C, H, W = x.size()
            x = x.view(B, T, C, H, W).contiguous()
            x = x[:,T//2,...]
            x_lst_new.append(x)
        x_lst_new =x_lst_new[-3:]
        if self.with_neck:
            x_lst_new = self.neck(x_lst_new)
        return x_lst_new


@MODELS.register_module()
class YOLOXDCNNeck(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 T=5,
                 use_xfeat = False
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.T = T
        self.use_xfeat = use_xfeat



    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        B,T,C,H,W = batch_inputs.size()
        batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
        x_lst = self.backbone(batch_inputs)
        x_lst_new =x_lst[-3:]
        if self.with_neck:
            x_lst_new,self.offsets_res = self.neck(x_lst_new)
        return x_lst_new

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        if self.use_xfeat:
            B, T, C, H, W = batch_inputs.size()
            batch_inputs2 = batch_inputs.view(B*T,C,H,W).contiguous()
            with torch.no_grad():
                out_batch = xfeat_inst.detectAndCompute(batch_inputs2/255, top_k=4096)
            match_res_batch = []
            for i in range(B):
                out_target=out_batch[i*T+(T-1)//2]
                match_res = []
                for t in range(T):
                    if t == (T-1)//2:
                        continue
                    out_t = out_batch[i*T+t]
                    with torch.no_grad():
                        idxs0, idxs1 = xfeat_inst.match(out_target['descriptors'], out_t['descriptors'], min_cossim=0.9)
                    mkpts_0, mkpts_1 = out_target['keypoints'][idxs0].cpu().numpy(), out_t['keypoints'][idxs1].cpu().numpy()
                    distances = np.sum((mkpts_0 - mkpts_1) ** 2, axis=1) ** 0.5
                    mkpts_0, mkpts_1 = mkpts_0[distances > 3], mkpts_1[distances > 3]
                    match_res.append([mkpts_0, mkpts_1])
                match_res_batch.append(copy.deepcopy(match_res))
            # self.show_match_res(batch_inputs,match_res_batch)
            with torch.no_grad():
                batch_offsets_gt = self.get_offset_gt(batch_inputs,match_res_batch)
            i=0
            for offset_pred,offset_gt in zip(self.offsets_res,batch_offsets_gt):
                offset_mask = offset_gt[1]
                offset_gt = offset_gt[0]
                if torch.sum(offset_mask)>0:
                    loss = torch.sum(torch.nn.functional.l1_loss(offset_pred,offset_gt,reduce=False)*offset_mask)/(torch.sum(offset_mask)+1e-10)
                    losses["loss_offset%01d"%i] = 0.01*loss
                else:
                    pass
                i+=1
        return losses

    def get_offset_gt(self,batch_inputs,match_res_batch):
        B, T, C, H, W = batch_inputs.size()

        disp_masks = torch.zeros((B, 2, T, H, W), dtype=torch.float)
        disp_masks = disp_masks.cuda()

        for b,match_res in enumerate(match_res_batch):
            for t in range(T):
                if t == (T - 1) // 2:
                    continue
                id = t if t< (T - 1) // 2 else t-1
                mkpts_0 = match_res_batch[b][id][0]  ##中心帧
                mkpts_1 = match_res_batch[b][id][1]  ##辅助帧
                offset_tmp = torch.from_numpy(np.array(mkpts_1)-np.array(mkpts_0)) ##以中心帧坐标为基准，加上offset，得到辅助帧的坐标
                offset_tmp = offset_tmp.permute(1,0).cuda()
                loations = np.array(mkpts_0,dtype=int)    ##这里为什么用辅助帧的位置呢？ 应该是写错了，这里应当是mkpts_0
                disp_masks[b,:,t,loations[:,1], loations[:,0]] = offset_tmp
        ##获得三种scales（1/8，1/16，1/32）特征下的offsets
        disp_masks0_lst,available_mask0_lst = self.get_offset_gt_singlescale(disp_masks,scale=8)
        disp_masks1_lst,available_mask1_lst = self.get_offset_gt_singlescale(disp_masks,scale=16)
        disp_masks2_lst,available_mask2_lst = self.get_offset_gt_singlescale(disp_masks,scale=32)
        return [disp_masks0_lst,available_mask0_lst],[disp_masks1_lst,available_mask1_lst],[disp_masks2_lst,available_mask2_lst]

    def get_offset_gt_singlescale(self,disp_masks,scale=8):
        # 用最大最小组合池化找出绝对值最大的值
        disp_masks0_0 = torch.nn.functional.max_pool3d(disp_masks, kernel_size=(1, scale, scale), stride=(1, scale, scale))
        disp_masks0_1 = -torch.nn.functional.max_pool3d(-disp_masks, kernel_size=(1, scale, scale), stride=(1, scale, scale))
        disp_masks0 = disp_masks0_0 * (torch.abs(disp_masks0_0) >= torch.abs(disp_masks0_1)) + \
                      disp_masks0_1 * (torch.abs(disp_masks0_1) >= torch.abs(disp_masks0_0))
        disp_masks0 = disp_masks0 / scale
        available_mask = torch.abs(disp_masks0[:, 0:1, ...] * disp_masks0[:, 1:, ...]) > 0.1
        available_mask = torch.cat([available_mask, available_mask], dim=1)

        disp_masks0_padded = torch.nn.functional.pad(disp_masks0, pad=(1, 1, 1, 1), value=0)
        available_mask_padded = torch.nn.functional.pad(available_mask, pad=(1, 1, 1, 1), value=0)
        disp_masks0_lst = torch.cat([
            disp_masks0_padded[..., 0:-2, 0:-2],  # 左上
            disp_masks0_padded[..., 0:-2, 1:-1],  # 中上
            disp_masks0_padded[..., 0:-2, 2:],  # 右上

            disp_masks0_padded[..., 1:-1, 0:-2],  # 左
            disp_masks0,  # 中
            disp_masks0_padded[..., 1:-1, 2:],  # 右

            disp_masks0_padded[..., 2:, 0:-2],  # 左下
            disp_masks0_padded[..., 2:, 1:-1],  # 中下
            disp_masks0_padded[..., 2:, 2:],  # 右下

        ], dim=1)
        available_mask_lst = torch.cat([
            available_mask_padded[..., 0:-2, 0:-2],  # 左上
            available_mask_padded[..., 0:-2, 1:-1],  # 中上
            available_mask_padded[..., 0:-2, 2:],  # 右上

            available_mask_padded[..., 1:-1, 0:-2],  # 左
            available_mask,  # 中
            available_mask_padded[..., 1:-1, 2:],  # 右

            available_mask_padded[..., 2:, 0:-2],  # 左下
            available_mask_padded[..., 2:, 1:-1],  # 中下
            available_mask_padded[..., 2:, 2:],  # 右下
        ], dim=1)
        return disp_masks0_lst,available_mask_lst


@MODELS.register_module()
class YOLOXC3D(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 C3D_cfg=None,
                 indep_bankbone=True,T=4
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        '''
        C3D_cfg包括一个列表，每个元素代表一个特征层次
        每个列表中又包括一个列表，每个元素代表一个网络层
        例如：
    C3D_cfg=[[{"type":"Conv3d","in_channels":128,"out_channels":128,"kernel_size":[3,3,3],"stride":[1,1,1],"padding":0},
             {"type":"Conv2d","in_channels":128,"out_channels":128,"kernel_size":[3,3],"stride":[1,1],"padding":1}],
             [{"type": "Conv3d", "in_channels": 128, "out_channels": 128, "kernel_size": [3, 3, 3], "stride": [1, 1, 1], "padding": 0},
              {"type": "Conv2d", "in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": [1, 1], "padding": 1}],
             [{"type": "Conv3d", "in_channels": 128, "out_channels": 128, "kernel_size": [3, 3, 3], "stride": [1, 1, 1], "padding": 0},
              {"type": "Conv2d", "in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": [1, 1], "padding": 1}]],
        '''
        self.indep_bankbone = indep_bankbone
        self.T = T
        self.C3D_cfg = C3D_cfg
        fusion_layers = []
        for level_cfg in self.C3D_cfg:
            level_lst= []
            for layer in level_cfg:
                func = getattr(torch.nn, layer["type"])
                del layer["type"]
                level_lst.append(func(**layer))
            fusion_layers.append(torch.nn.Sequential(*level_lst))
        self.fusion_layers =torch.nn.ModuleList(fusion_layers)
        pass

    def loss(self, batch_inputs: Tensor,batch_data_samples: SampleList) -> Union[dict, list]:
        if os.environ['LOCAL_RANK']=="0" and random.random()>0:
            self.random_show(data= batch_inputs,batch_data_samples=batch_data_samples)
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def random_show(self,data,batch_data_samples):
        def draw_rectes(img,rectes):
            img = img.copy()
            for rect in rectes:
                img= cv2.rectangle(img,pt1=[int(rect[0]),int(rect[1])],pt2=[int(rect[2]),int(rect[3])],color=(0,0,255),thickness=2)
            return img
        if os.path.exists("00show_train") is False:
            os.makedirs("00show_train")
        name = "00show_train/%05d.jpg"%random.randint(0,1000)
        bboxes = batch_data_samples[0].gt_instances.bboxes.data.cpu().numpy()
        labels = batch_data_samples[0].gt_instances.labels.data.cpu().numpy()
        if len(data.size())==4:
            data = data.unsqueeze(1)
        data = data[0,...].data.cpu().numpy()
        data = np.transpose(data,(0,2,3,1)) ##[T,3,H,W] -->[T,H,W,3]
        # data =data[:,:,:,[2,1,0]] ##rgb -->bgr
        data = (data-data.min())*255/(data.max()-data.min())
        data = np.array(data,dtype=np.uint8)
        img_show = data[(data.shape[0]-1)//2]
        img_show = draw_rectes(img_show,bboxes)
        data[(data.shape[0] - 1) // 2] = img_show
        for i in range(data.shape[0]):
            cv2.imwrite(name.replace('.jpg','--%03d.jpg'%i),data[i])

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        B,T,C,H,W = batch_inputs.size()
        batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
        x_lst = self.backbone(batch_inputs)
        x_lst_new = []
        for i, x in enumerate(x_lst):
            BT, C, H, W = x.size()
            x = x.view(B , T, C, H, W).contiguous()
            x = x.permute(0,2,1,3,4).contiguous()
            x = self.fusion_layers[i](x)
            x = x.squeeze(2)
            x_lst_new.append(x)

        if self.with_neck:
            x_lst_new = self.neck(x_lst_new)
        return x_lst_new


@MODELS.register_module()
class YOLOXC3DRes(YOLOXC3D):
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        if self.indep_bankbone:
            B, T, C, H, W = batch_inputs.size()
            batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                BT, C, H, W = x.size()
                x = x.view(B , T, C, H, W).contiguous()
                x_target = x[:,(T-1)//2,:,:,:].clone()
                x = x.permute(0,2,1,3,4).contiguous()
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x_target+x)
        else:
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                x_target = x.clone()
                x = x.unsqueeze(2)
                x = x.repeat((1, 1, self.T, 1, 1)).contiguous()
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x + x_target)

        if self.with_neck:
            x_lst_new = self.neck(x_lst_new)
        return x_lst_new


@MODELS.register_module()
class YOLOXDCNRes(YOLOXC3DRes,SingleStageDetector):
    '''超算跑不通'''
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 C3D_cfg=None,
                 indep_bankbone=True,T=4,
                 DCN_cfg=None
                 ) -> None:
        SingleStageDetector.__init__(
            self,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.indep_bankbone = indep_bankbone
        self.T = T
        self.C3D_cfg = C3D_cfg
        self.DCN_cfg = DCN_cfg

        fusion_layers = []
        for level_cfg in self.C3D_cfg:
            level_lst= []
            for layer in level_cfg:
                if layer["type"] in C3DCNs:
                    func = globals()[layer["type"]]
                else:
                    func = getattr(torch.nn, layer["type"])
                del layer["type"]
                level_lst.append(func(**layer))
            fusion_layers.append(torch.nn.Sequential(*level_lst))
        self.fusion_layers =torch.nn.ModuleList(fusion_layers)


        dcn_layers = []
         ##一个3D-DCN一个Sequential
        for level_cfg in self.DCN_cfg:
            one_dcn =[]
            dcn_main =  globals()[level_cfg["main"]["type"]]
            del level_cfg["main"]["type"]
            dcn_main = dcn_main(**level_cfg["main"])
            one_dcn.append(dcn_main)
            offset_layers = []
            for layer in level_cfg["offset_cfg"]:
                func = getattr(torch.nn, layer["type"])
                del layer["type"]
                offset_layers.append(func(**layer))
            one_dcn.append(torch.nn.Sequential(*offset_layers))
            dcn_layers.append(torch.nn.ModuleList(one_dcn))
        self.dcn_layers=torch.nn.ModuleList(dcn_layers)
        pass

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        if self.indep_bankbone:
            B, T, C, H, W = batch_inputs.size()
            batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                BT, C, H, W = x.size()
                x = x.view(B , T, C, H, W).contiguous()
                x_target = x[:,(T-1)//2,:,:,:].clone()
                x = x.permute(0,2,1,3,4).contiguous()
                ##DCN
                offset = self.dcn_layers[i][1](x)
                x = self.dcn_layers[i][0](x,offset)
                ##fusion
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x_target+x)
        else:
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                x_target = x.clone()
                x = x.unsqueeze(2)
                x = x.repeat((1, 1, self.T, 1, 1)).contiguous()
                ##DCN
                offset = self.dcn_layers[i][1](x)
                x = self.dcn_layers[i][0](x,offset)
                ##fusion
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x + x_target)

        if self.with_neck:
            x_lst_new = self.neck(x_lst_new)


        return x_lst_new


@MODELS.register_module()
class YOLOXDCNResXFeat(YOLOXC3DRes,SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 C3D_cfg=None,
                 indep_bankbone=True,T=4,
                 DCN_cfg=None,
                 use_xfeat = False
                 ) -> None:
        SingleStageDetector.__init__(
            self,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.indep_bankbone = indep_bankbone
        self.T = T
        self.C3D_cfg = C3D_cfg
        self.DCN_cfg = DCN_cfg
        self.use_xfeat = use_xfeat
        # self.xfeat = XFeat()
        # self.xfeat.eval()

        fusion_layers = []
        for level_cfg in self.C3D_cfg:
            level_lst= []
            for layer in level_cfg:
                if layer["type"] in C3DCNs:
                    func = globals()[layer["type"]]
                else:
                    func = getattr(torch.nn, layer["type"])
                del layer["type"]
                level_lst.append(func(**layer))
            fusion_layers.append(torch.nn.Sequential(*level_lst))
        self.fusion_layers =torch.nn.ModuleList(fusion_layers)


        dcn_layers = []
         ##一个3D-DCN一个Sequential
        for level_cfg in self.DCN_cfg:
            one_dcn =[]
            dcn_main =  globals()[level_cfg["main"]["type"]]
            del level_cfg["main"]["type"]
            dcn_main = dcn_main(**level_cfg["main"])
            one_dcn.append(dcn_main)
            offset_layers = []
            for layer in level_cfg["offset_cfg"]:
                func = getattr(torch.nn, layer["type"])
                del layer["type"]
                offset_layers.append(func(**layer))
            one_dcn.append(torch.nn.Sequential(*offset_layers))
            dcn_layers.append(torch.nn.ModuleList(one_dcn))
        self.dcn_layers=torch.nn.ModuleList(dcn_layers)
        pass


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        self.offsets_res = []
        if self.indep_bankbone:
            B, T, C, H, W = batch_inputs.size()
            batch_inputs = batch_inputs.view(B*T,C,H,W).contiguous()
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                BT, C, H, W = x.size()
                x = x.view(B , T, C, H, W).contiguous()
                x_target = x[:,(T-1)//2,:,:,:].clone()
                x = x.permute(0,2,1,3,4).contiguous()
                ##DCN
                offset = self.dcn_layers[i][1](x)
                x = self.dcn_layers[i][0](x,offset)
                ##fusion
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x_target+x)
                self.offsets_res.append(offset)
        else:
            x_lst = self.backbone(batch_inputs)
            x_lst_new = []
            for i, x in enumerate(x_lst):
                x_target = x.clone()
                x = x.unsqueeze(2)
                x = x.repeat((1, 1, self.T, 1, 1)).contiguous()
                ##DCN
                offset = self.dcn_layers[i][1](x)
                x = self.dcn_layers[i][0](x,offset)
                ##fusion
                x = self.fusion_layers[i](x)
                x = x.squeeze(2)
                x_lst_new.append(x + x_target)
                self.offsets_res.append(offset)

        if self.with_neck:
            x_lst_new = self.neck(x_lst_new)

        return x_lst_new

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        if self.use_xfeat:
            B, T, C, H, W = batch_inputs.size()
            batch_inputs2 = batch_inputs.view(B*T,C,H,W).contiguous()
            with torch.no_grad():
                out_batch = xfeat_inst.detectAndCompute(batch_inputs2/255, top_k=4096)
            match_res_batch = []
            for i in range(B):
                out_target=out_batch[i*T+(T-1)//2]
                match_res = []
                for t in range(T):
                    if t == (T-1)//2:
                        continue
                    out_t = out_batch[i*T+t]
                    with torch.no_grad():
                        idxs0, idxs1 = xfeat_inst.match(out_target['descriptors'], out_t['descriptors'], min_cossim=0.9)
                    mkpts_0, mkpts_1 = out_target['keypoints'][idxs0].cpu().numpy(), out_t['keypoints'][idxs1].cpu().numpy()
                    distances = np.sum((mkpts_0 - mkpts_1) ** 2, axis=1) ** 0.5
                    mkpts_0, mkpts_1 = mkpts_0[distances > 3], mkpts_1[distances > 3]
                    match_res.append([mkpts_0, mkpts_1])
                match_res_batch.append(copy.deepcopy(match_res))
            # self.show_match_res(batch_inputs,match_res_batch)
            with torch.no_grad():
                batch_offsets_gt = self.get_offset_gt(batch_inputs,match_res_batch)
            i=0
            for offset_pred,offset_gt in zip(self.offsets_res,batch_offsets_gt):
                offset_mask = offset_gt[1]
                offset_gt = offset_gt[0]
                if torch.sum(offset_mask)>0:
                    loss = torch.sum(torch.nn.functional.l1_loss(offset_pred,offset_gt,reduce=False)*offset_mask)/(torch.sum(offset_mask)+1e-10)
                    losses["loss_offset%01d"%i] = 0.01*loss
                else:
                    pass
                i+=1
        return losses

    def get_offset_gt(self,batch_inputs,match_res_batch):
        B, T, C, H, W = batch_inputs.size()

        disp_masks = torch.zeros((B, 2, T, H, W), dtype=torch.float)
        disp_masks = disp_masks.cuda()

        for b,match_res in enumerate(match_res_batch):
            for t in range(T):
                if t == (T - 1) // 2:
                    continue
                id = t if t< (T - 1) // 2 else t-1
                mkpts_0 = match_res_batch[b][id][0]  ##中心帧
                mkpts_1 = match_res_batch[b][id][1]  ##辅助帧
                offset_tmp = torch.from_numpy(np.array(mkpts_1)-np.array(mkpts_0)) ##以中心帧坐标为基准，加上offset，得到辅助帧的坐标
                offset_tmp = offset_tmp.permute(1,0).cuda()
                loations = np.array(mkpts_0,dtype=int)    ##这里为什么用辅助帧的位置呢？ 应该是写错了，这里应当是mkpts_0
                disp_masks[b,:,t,loations[:,1], loations[:,0]] = offset_tmp
        ##获得三种scales（1/8，1/16，1/32）特征下的offsets
        disp_masks0_lst,available_mask0_lst = self.get_offset_gt_singlescale(disp_masks,scale=8)
        disp_masks1_lst,available_mask1_lst = self.get_offset_gt_singlescale(disp_masks,scale=16)
        disp_masks2_lst,available_mask2_lst = self.get_offset_gt_singlescale(disp_masks,scale=32)
        return [disp_masks0_lst,available_mask0_lst],[disp_masks1_lst,available_mask1_lst],[disp_masks2_lst,available_mask2_lst]

    def get_offset_gt_singlescale(self,disp_masks,scale=8):
        # 用最大最小组合池化找出绝对值最大的值
        disp_masks0_0 = torch.nn.functional.max_pool3d(disp_masks, kernel_size=(1, scale, scale), stride=(1, scale, scale))
        disp_masks0_1 = -torch.nn.functional.max_pool3d(-disp_masks, kernel_size=(1, scale, scale), stride=(1, scale, scale))
        disp_masks0 = disp_masks0_0 * (torch.abs(disp_masks0_0) >= torch.abs(disp_masks0_1)) + \
                      disp_masks0_1 * (torch.abs(disp_masks0_1) >= torch.abs(disp_masks0_0))
        disp_masks0 = disp_masks0 / scale
        available_mask = torch.abs(disp_masks0[:, 0:1, ...] * disp_masks0[:, 1:, ...]) > 0.1
        available_mask = torch.cat([available_mask, available_mask], dim=1)

        disp_masks0_padded = torch.nn.functional.pad(disp_masks0, pad=(1, 1, 1, 1), value=0)
        available_mask_padded = torch.nn.functional.pad(available_mask, pad=(1, 1, 1, 1), value=0)
        disp_masks0_lst = torch.cat([
            disp_masks0_padded[..., 0:-2, 0:-2],  # 左上
            disp_masks0_padded[..., 0:-2, 1:-1],  # 中上
            disp_masks0_padded[..., 0:-2, 2:],  # 右上

            disp_masks0_padded[..., 1:-1, 0:-2],  # 左
            disp_masks0,  # 中
            disp_masks0_padded[..., 1:-1, 2:],  # 右

            disp_masks0_padded[..., 2:, 0:-2],  # 左下
            disp_masks0_padded[..., 2:, 1:-1],  # 中下
            disp_masks0_padded[..., 2:, 2:],  # 右下

        ], dim=1)
        available_mask_lst = torch.cat([
            available_mask_padded[..., 0:-2, 0:-2],  # 左上
            available_mask_padded[..., 0:-2, 1:-1],  # 中上
            available_mask_padded[..., 0:-2, 2:],  # 右上

            available_mask_padded[..., 1:-1, 0:-2],  # 左
            available_mask,  # 中
            available_mask_padded[..., 1:-1, 2:],  # 右

            available_mask_padded[..., 2:, 0:-2],  # 左下
            available_mask_padded[..., 2:, 1:-1],  # 中下
            available_mask_padded[..., 2:, 2:],  # 右下
        ], dim=1)
        return disp_masks0_lst,available_mask_lst


    def show_match_res(self,batch_inputs,match_res_batch):
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        B, T, C, H, W = batch_inputs.size()
        show_imgs = batch_inputs.data.cpu().numpy()
        show_imgs = np.array(show_imgs,dtype=np.uint8)
        show_imgs = show_imgs.transpose(0,1,3,4,2)
        for b,match_res in enumerate(match_res_batch):
            img_target = show_imgs[b,(T-1)//2,...].copy()
            for t in range(T):
                if t == (T - 1) // 2:
                    continue
                img_curr = show_imgs[b,t,...].copy()
                id = t if t< (T - 1) // 2 else t-1
                mkpts_0 = match_res_batch[b][id][0]
                mkpts_1 = match_res_batch[b][id][1]
                show_matches(img_target, img_curr, mkpts_0, mkpts_1, rate=1)
        pass


def show_matches(img1,img2,mkpts1,mkpts2,rate=0.1):
    ids = range(mkpts1.shape[0])
    if mkpts1.shape[0]>100:
        showN = max(1,int(mkpts1.shape[0]*rate)-1)
        ids = random.sample(ids,showN)
    mkpts1, mkpts2 = mkpts1[ids].astype(int),mkpts2[ids].astype(int)
    mkpts2[:,1] +=img1.shape[0]
    img = np.concatenate([img1,img2],axis=0)
    for i in range(mkpts1.shape[0]):
        img = cv2.line(img,pt1=mkpts1[i],pt2=mkpts2[i],color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),thickness=1)
    cv2.imshow("img",img)
    cv2.waitKey(0)


class Pseudo3DDCN(torch.nn.Module):
    r"""
    伪的3D-DCN, offset只能再W,H 两个维度上
    输入 feat 与 offset
    feat: [B, C, T, H, W]
    offset:[B, K, T, H, W]
    其中 K =kernel_size[0]*kernel_size[1]*2
    """
    def __init__(self,in_channels, out_channels,T, kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.deformconv2d = mmcv.ops.deform_conv.DeformConv2d(
            in_channels=in_channels*T,
            out_channels=out_channels*T,
            kernel_size=kernel_size,
            stride=stride,padding=padding,bias=False,
            groups=T,
            deform_groups=T,
        )
        self.in_channels, self.out_channels, self.T = in_channels, out_channels,T


    def forward(self,feat,offset):
        feat = feat.permute(0,2,1,3,4).contiguous()  #: [B, T, C, H, W]
        B, T, C, H, W = feat.size()
        feat = feat.view(B,T*C,H,W).contiguous() #: [B,T*C, H, W]

        offset = offset.permute(0,2,1,3,4).contiguous()  #: [B, T, K, H, W]
        _, _, K, _, _ = offset.size()
        offset = offset.view(B,T*K,H,W).contiguous() #: [B,T*K, H, W]
        feat = self.deformconv2d.forward(x=feat,offset=offset)
        feat = feat.view(B,T,C,H,W).contiguous()
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()  #: [B, C,T, H, W]
        return feat
