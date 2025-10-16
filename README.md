# LwST-FSD
Pytorch implementation of the paper "LwST-FSD: Efficient Forest Smoke Detection via Lightweight Spatio-temporal Modeling"

## Introduction
- **ST-CSPDarknet**
![ST-CSPLayer]( https://ars.els-cdn.com/content/image/1-s2.0-S0020025525006139-gr004_lrg.jpg)
To address the computational limitations of existing spatio-temporal modeling approaches, we adopt the Temporal Shift Module (TSM) - a widely-recognized efficient temporal modeling technique in video understanding tasks. By integrating TSM into the lightweight 2D convolutional network CSPDarkNet, we propose a novel 3D spatio-temporal backbone network called TS-CSPDarkNet. This architecture achieves efficient spatio-temporal feature fusion without introducing additional parameters or computational overhead, overcoming the trade-off between accuracy and efficiency in wildfire smoke detection.

- **PTANeck**
![PTANeck](https://ars.els-cdn.com/content/image/1-s2.0-S0020025525006139-gr005_lrg.jpg)
To address three critical challenges in downstream detection: (1) temporal dimension reduction through asymmetric frame processing between target and support frames, (2) multi-scale spatial feature fusion, and (3) channel alignment, we propose a novel Partial Temporal Aggregation Neck (PTANeck) structure to produce unified spatio-temporal features.

- **SKLFS-WildFire2025 dataset**
![Key Characteristics of SKLFS-WildFire2025. ](https://ars.els-cdn.com/content/image/1-s2.0-S0020025525006139-gr006_lrg.jpg)
SKLFS-WildFire2025 is a newly released dataset for smoke detection, representing a significant advancement in terms of both dataset scale and diversity. It contains 8,363 video clips (totaling 152,700 annotated frames) and 48,777 smoke-affected images extracted from 3,952 video segments, all reserved exclusively for testing.


## How to Install
The installation is exactly the same as [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection). This project is constructed based on the [mmdetection-V3.2.0](https://github.com/open-mmlab/mmdetection/tree/3.x)

⚠ Before using the code from this project, it is assumed that you are already very familiar with the mmdetection codebase. 

⚠If you encounter errors caused by references unrelated to reproducing the methods, they may be remnants from earlier exploratory experiments and can be safely removed.

After installing mmdetection, users can copy the files and folders of this project to the mmdetection directory and replace files with the same name. 
Our changes to native mmdetection are as follows:
```
mmdetection-master
├── mmdet
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── VID_Dataset_ChongWang.py (MultiFrameDatasetSmoke)
│   │   ├── transforms
│   │   │   ├── __init__.py
│   │   │   ├── transforms.py (RandomAffineMultiImages,YOLOXHSVRandomAugMultiImages,RandomFlipMultiImages,ResizeMultiImages,PadMultiImages,MosaicMultiImages,MixUpMultiImages)
│   │   │   ├── formatting.py (PackDetInputsMultiImages)
│   ├── models
│   │   ├──  data_preprocessors
│   │   │   ├── __init__.py
│   │   │   ├── data_preprocessor.py (DetDataPreprocessorMultiImages,BatchSyncRandomResizeMultiImages)
│   │   ├── detectors
│   │   │   ├── __init__.py
│   │   │   ├── yolox_VID.py (YOLOXTSM_PTA)
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   ├── csp_darknet_tsm.py (CSPDarknetTSM)
│   │   ├── necks
│   │   │   ├── __init__.py
│   │   │   ├── yolox_pafpn_PTA.py (YOLOXPAFPN_PTA)
├── ConfigsVIDSmoke
```
This GitHub repository does not contain complete, runnable code. Only the key files mentioned above are provided. Those who wish to reproduce the results can make modifications based on the mmdetection framework. Register the custom classes using the __init__.py file in each folder. Only focus on the custom classes within parentheses, as they are essential for reproducing the results of this paper. Classes not marked within parentheses may cause errors in the code, and irrelevant code should be deleted.

## Data Preparation
- **FIgLib**

  Organize the dataset in the format of VOC 2007. Each subset contains a JPEGImages folder for storing images, Annotations for storing XML annotation files, and a txt format sample list
```
FIgLib
├── train
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_train.txt
├── val
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_val.txt
├── test
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_test.txt
```
- **SKLFS-WildFire**
  [SKLFS-WildFire dataset download here (unavailable)](https://xxx) 
  Same as FIgLib. However, SKLFS-WildFire does not set a validation set, and users can self split from the training set
```
SKLFS-WildFire
├── train
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire_train.txt
├── test
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire_test.txt
```

- **SKLFS-WildFire2025**
  [SKLFS-WildFire2025 dataset download here (unavailable)](https://xxx) 
  Same as FIgLib. However, SKLFS-WildFire2025 does not set a validation set, and users can self split from the training set
```
SKLFS-WildFire2025
├── train
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire2025_train.txt
├── test
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire2025_test.txt
```

## Getting Started

   For training, run
  ```Shell
  python tools/train.py [path_to_your_config] 
  ```



## Finetuned Models

| Model              |  description                    | Dateset      | Download     |
|--------------------|---------------------------------|--------------|--------------|
| LwST-FSD           | 5 frames TS-CSPDarkNet-Small    |FIgLib        |[download](https://pan.ustc.edu.cn/share/index/5b015f2c73e24fd48b82) |


## Citation

If you use this code or dataset in your research, please cite this project.

```
@article{WANG2025122481,
title = {LwST-FSD: Efficient forest smoke detection via lightweight spatio-temporal modeling},
journal = {Information Sciences},
volume = {719},
pages = {122481},
year = {2025},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2025.122481},
author = {Chong Wang and Hai Song and Zhong Wang and Zhilin Shan and Qixing Zhang},
}
```

