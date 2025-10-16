# Copyright (c) OpenMMLab. All rights reserved.
import copy,json,random,sys,cv2
import os.path
import os.path as osp
from typing import List, Union

import numpy as np
import torch.utils.data
import tqdm
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset
from tqdm.contrib.concurrent import process_map

from torch.utils.data import Dataset
from mmengine.dataset.base_dataset import Compose
from mmdet.structures.bbox import (BaseBoxes, HorizontalBoxes, bbox2distance,
                                   distance2bbox, get_box_tensor)
from .coco import CocoDataset



@DATASETS.register_module()
class SingleFrameDataset(BaseDetDataset):
    METAINFO = {
        'classes': ('airplane', 'antelope', 'bear', 'bicycle', 'bird',
                    'bus', 'car', 'cattle', 'dog', 'domestic cat',
                    'elephant', 'fox', 'giant panda', 'hamster', 'horse',
                    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
                    'red panda', 'sheep', 'snake', 'squirrel', 'tiger',
                    'train', 'turtle', 'watercraft', 'whale', 'zebra'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
                    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164)]
    }

    def __init__(self, annotation_files: list,
                 pipeline: list,
                 test_mode: bool = False,
                 metainfo = None,
                 max_refetch: int = 1000,
                 lazy_init: bool = False,
                 return_classes=False):
        self.lazy_init =lazy_init
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.max_refetch=max_refetch
        self.test_mode = test_mode
        self.annotation_files=annotation_files
        self.full_init()
        self.pipeline = Compose(pipeline)
        self.return_classes=return_classes


    def __getitem__(self, idx: int) -> dict:
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                print(self.data_list[idx],"-----------------read data %09d Failed -----------------"%idx)
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def prepare_data(self, idx) :
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def get_data_info(self, idx: int):
        """
        Returns:
            dict: The idx-th annotation of the dataset.
        """
        data_info_org = copy.deepcopy(self.data_list[idx])
        data_info = {}
        data_info['sample_idx'] = idx
        data_info['img_path'] = data_info_org["img_path"]
        data_info['img_id'] = idx
        data_info['seg_map_path'] = None
        data_info['instances'] = [{"ignore_flag":0,"bbox":bbox,'bbox_label':label} \
                                  for bbox,label in zip(data_info_org["bboxes"],data_info_org["labels"])]
        if hasattr(self,"return_classes"):
            if self.return_classes:
                data_info['text'] = self.metainfo['classes']
                data_info['custom_entities'] = True
        return data_info

    def load_data_list(self) -> List[dict]:
        # load and parse data_infos.
        data_list = []
        for file in self.annotation_files:
            with open(file,'r') as f:
                data_list.extend(json.load(f))
        return data_list

    def full_init(self) -> None:
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        self._fully_initialized = True

    def __len__(self) -> int:
        return len(self.data_list)


@DATASETS.register_module()
class SingleFrameDatasetSmoke(SingleFrameDataset):
    METAINFO = {
        'classes': ('smoke'),
        'palette': [(220, 20, 60)]
    }

    def __init__(self, *args,**kwargs):
        SingleFrameDataset.__init__(self,*args,**kwargs)
        # self.video_dict = self.obtain_video_dict_accordname()
        # self.data_statistic()
        pass


    def obtain_video_dict_accordname(self):
        video_dict={}
        for i,name in enumerate(self.data_list):
            name = '--'.join(name['img_path'].split('--')[:-1])
            if name not in video_dict.keys():
                video_dict[name] = [i]
            else:
                video_dict[name].append(i)
        return video_dict

    def data_statistic(self):
        annotations = [np.array(data["labels"]) for data in self.data_list]

        label_smoke = np.array([np.sum(ann==0) for ann in annotations],dtype=int)
        label_smoke = np.array((label_smoke>=1)*1,dtype=int)

        gt_video_fire, gt_video_smoke = [], []
        for video in self.video_dict.keys():
            ids = np.array(self.video_dict[video])
            gt_smoke = np.max(label_smoke[ids])
            gt_video_smoke.append(gt_smoke)
        gt_video_smoke = np.array(gt_video_smoke, dtype=int)

        print(
            "****   Total Images: %08d       ImagesWithSmoke: %08d    ImagesNeg: %08d  ********" % (
                len(annotations),  np.sum(label_smoke), len(annotations) - np.sum((( label_smoke) >= 1) * 1)
            ))

        print(
            "****   Total Videos: %08d      VideosWithSmoke: %08d        VideosNeg: %08d  ********" % (
                len(self.video_dict), np.sum(gt_video_smoke),   len(self.video_dict) - np.sum(((gt_video_smoke) >= 1) * 1)
            ))

@DATASETS.register_module()
class MultiFrameDataset(SingleFrameDataset):
    def __init__(self, annotation_files: list,
                 related_ids:list,
                 pipeline,
                 test_mode: bool = False,
                 metainfo = None,
                 max_refetch: int = 1000,
                 lazy_init:bool=False
                 ):
        ''''''
        self.lazy_init =lazy_init
        files_str  = ','.join(annotation_files)
        if "COCO" in files_str or "coco" in files_str:
            self.METAINFO = CocoDataset.METAINFO

        self.pipeline = Compose(pipeline)
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.related_ids=related_ids
        self.max_refetch=max_refetch
        self.test_mode = test_mode
        self.annotation_files=annotation_files
        self.full_init()
        self.get_support_pathes()
        pass

    def get_support_pathes(self):
        self.support_lst = []
        for data in tqdm.tqdm(self.data_list):
            img_path = data["img_path"]
            if "DET" in img_path or "COCO" in img_path:
                self.support_lst.append([None for i in range(len(self.related_ids))])
            elif "VID" in img_path:
                tmp_pathes = []
                img_path, img_name = os.path.split(img_path)
                frame_id = int(img_name.split('.')[0])
                for related_id in self.related_ids:
                    frame_id_curr = frame_id + related_id
                    support_img_path = os.path.join(img_path, "%06d.JPEG" % frame_id_curr)
                    if os.path.exists(support_img_path):
                        tmp_pathes.append(support_img_path)
                    else:
                        tmp_pathes.append(None)
                self.support_lst.append(copy.deepcopy(tmp_pathes))
            else:
                raise ("img_path error:%s" % img_path)


    def prepare_data(self, idx) :
        try:
            data_info = self.get_data_info(idx)
            img_path = data_info["img_path"]
            img = cv2.imread(img_path)
            data_info["img"] = img
            support_imgs = [cv2.imread(patha) if patha else img.copy() for patha in self.support_lst[idx]]
            for i,support_img in enumerate(support_imgs):
                data_info["img_%04d"%i] = support_img
            h,w,c = img.shape
            data_info["img_shape"] = (h,w)
            data_info["ori_shape"] = (h,w)
            bboxes =np.array([v["bbox"] for v in data_info["instances"]],dtype=np.float32)
            data_info["gt_bboxes"] = HorizontalBoxes(bboxes)
            data_info["gt_bboxes_labels"] = np.array([v['bbox_label'] for v in data_info["instances"]],dtype=np.int64)
            data_info['gt_ignore_flags'] = np.array([False]*len(data_info["instances"]),dtype=bool)
            res =  self.pipeline(data_info)
            return res
        except Exception as e:
            print(e)
            return None


@DATASETS.register_module()
class MultiFrameDatasetSmoke(MultiFrameDataset,SingleFrameDatasetSmoke):
    METAINFO = {
        'classes': ('smoke'),
        'palette': [(220, 20, 60)]
    }
    def __init__(self, *args,**kwargs):
        MultiFrameDataset.__init__(self,*args,**kwargs)
        self.video_dict = SingleFrameDatasetSmoke.obtain_video_dict_accordname(self)
        SingleFrameDatasetSmoke.data_statistic(self)

    def get_support_pathes(self):
        # self.data_list = self.data_list[0:3000]


        self.support_lst = []
        '''按照 目录-Video 作为key, frame_id_lst 作为value'''
        frames_dict = {}
        for data in tqdm.tqdm(self.data_list):
            img_path = data["img_path"]
            video_path = '--'.join(img_path.split('--')[:-1])
            frame_id = img_path.split('--')[-1].replace(".jpg",'').replace(".png",'').replace(".jpeg",'')
            frame_id = int(frame_id)
            if video_path not in frames_dict.keys():
                frames_dict[video_path] = [frame_id]
            else:
                frames_dict[video_path].append(frame_id)
        print('get frame ids Successfully')
        for key in frames_dict.keys():
            frames_dict[key].sort()
        print('sort frame ids Successfully')

        for data in tqdm.tqdm(self.data_list):
            img_path = data["img_path"]
            video_path = '--'.join(img_path.split('--')[:-1])
            frame_id = img_path.split('--')[-1].replace(".jpg",'')
            frame_id = int(frame_id)
            frame_id_lst = frames_dict[video_path]
            lst_id = frame_id_lst.index(frame_id)
            tmp_pathes = []
            for  related_id in self.related_ids:
                id_curr = lst_id+related_id
                if id_curr>=0 and id_curr<len(frame_id_lst):
                    path_curr = video_path + "--%05d.jpg"%frame_id_lst[lst_id+related_id]
                    tmp_pathes.append(copy.deepcopy(path_curr))
                else:
                    tmp_pathes.append(None)
            self.support_lst.append(copy.deepcopy(tmp_pathes))

@DATASETS.register_module()
class MultiFrameDatasetFIgLib(MultiFrameDataset):
    METAINFO = {
        'classes': ('smoke'),
        'palette': [(220, 20, 60)]
    }


    def get_support_pathes(self):
        # self.data_list = self.data_list[0:3000]
        self.support_lst = []
        '''按照 目录-Video 作为key, frame_id_lst 作为value'''
        frames_dict = {}
        for data in tqdm.tqdm(self.data_list):
            img_path = data["img_path"]
            video_path = '--'.join(img_path.split('--')[:-1])
            t, frame_id = img_path.split('---')[-1].replace(".jpg", '').replace(".png", '').replace(".jpeg", '').split("_")
            frame_id = int(frame_id)
            if video_path not in frames_dict.keys():
                frames_dict[video_path] = [[t, frame_id]]
            else:
                frames_dict[video_path].append([t, frame_id])
        print('get frame ids Successfully')
        for key in frames_dict.keys():
            frames_dict[key] = sorted(frames_dict[key], key=lambda x: x[1])
        print('sort frame ids Successfully')

        for data in tqdm.tqdm(self.data_list):
            img_path = data["img_path"]
            video_path = '--'.join(img_path.split('--')[:-1])
            t, frame_id = img_path.split('---')[-1].replace(".jpg", '').replace(".png", '').replace(".jpeg", '').split("_")
            frame_id = int(frame_id)
            frame_id_lst = [v[1] for v in frames_dict[video_path]]
            lst_id = frame_id_lst.index(frame_id)
            tmp_pathes = []
            for  related_id in self.related_ids:
                id_curr = lst_id+related_id
                if id_curr>=0 and id_curr<len(frame_id_lst):
                    if frame_id_lst[lst_id + related_id] >= 0:
                        path_curr = video_path + "---%s_+%05d.jpg" % (
                        frames_dict[video_path][lst_id + related_id][0], frame_id_lst[lst_id + related_id])
                    else:
                        path_curr = video_path + "---%s_%06d.jpg" % (
                        frames_dict[video_path][lst_id + related_id][0], frame_id_lst[lst_id + related_id])
                    tmp_pathes.append(copy.deepcopy(path_curr))
                else:
                    tmp_pathes.append(None)
            self.support_lst.append(copy.deepcopy(tmp_pathes))


@DATASETS.register_module()
class MultiFrameDatasetCOCO(MultiFrameDataset):
    METAINFO = CocoDataset.METAINFO
