# Copyright (c) iFireTEK (Chong Wang). All rights reserved.
import os.path

from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK
from onnxconverter_common import float16
from onnx import load_model, save_model

def convert2fp16(onnx_name,save_name):
    onnx_model = load_model(onnx_name)
    print(onnx_model.graph)
    model_fp16 = float16.convert_float_to_float16(onnx_model,keep_io_types=False)
    # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
    save_model(model_fp16, save_name)


# img = r'E:\fire-data\00shengpingtai\shengpingtai_test\Smoke\JPEGImages33\26c6c72dd81e40c8a8a4f7c7a7d85102_2023-04-14---18--55--27.241076--00095_fb.jpg'
# work_dir = r'E:/mmdetection320/work_dirs/vitdet_faster-rcnn_vit-b-mae_lsj-100e/'
# save_file = 'iter_427920.onnx'
# deploy_cfg = 'E:/mmdeploy-main/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
# model_cfg = r'E:\mmdetection320\ConfigsWildFire\ViTDet\vitdet_faster-rcnn_vit-b-mae_lsj-100e.py'
# model_checkpoint = r'E:\mmdetection320\work_dirs\vitdet_faster-rcnn_vit-b-mae_lsj-100e\iter_427920.pth'
# device = 'cpu'
img = r'E:\fire-data\saida-testvideos\JPEGImages\20250325145500-20250325145600_1--00094.jpg'
work_dir = r'E:\mmdetection320\work_dir_VIDSmoke\yolox_s_8xb8-300e_cocopretrained'
deploy_cfg = 'E:\mmdeploy-main\configs_CW\detection_onnxruntime-fp16_dynamic.py'
model_cfg = r'E:\mmdetection320\ConfigsVIDSmoke\yolox_s_8xb8-300e_cocopretraind.py'
model_checkpoint = r'E:\mmdetection320\work_dir_VIDSmoke\yolox_s_8xb8-300e_cocopretrained\epoch_28.pth'
save_file = 'epoch_28.onnx'
save_file_fp16 = os.path.join(work_dir,save_file.replace('.onnx','_fp16.onnx'))


device = 'cpu'
# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)

convert2fp16(onnx_name=os.path.join(work_dir,save_file),save_name=save_file_fp16)