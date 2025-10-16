import torch
import pickle
import os


def transfer_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to('cpu')
    elif isinstance(obj, dict):
        return {k: transfer_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [transfer_to_cpu(elem) for elem in obj]
    else:
        return obj


# 输入pkl文件列表
file_list = [
    r'E:\mmdetection320\work_dir_VIDSmoke\yolo11\test_results.pkl',
    r'E:\mmdetection320\work_dir_VIDSmoke\yolo11\val_results.pkl',
    r'E:\mmdetection320\work_dir_VIDSmoke\yolo12\test_results.pkl',
    r'E:\mmdetection320\work_dir_VIDSmoke\yolo12\val_results.pkl',

]

for file_path in file_list:
    # 加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        data = transfer_to_cpu(data)

    # 生成新文件名
    new_file_path = os.path.splitext(file_path)[0] + '_cpu.pkl'

    # 保存数据
    with open(new_file_path, 'wb') as f:
        pickle.dump(data, f)