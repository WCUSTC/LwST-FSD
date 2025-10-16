from .coco import CocoDataset
from mmdet.registry import DATASETS



@DATASETS.register_module()
class FireSmokeDataset(CocoDataset):
    METAINFO = {
        'classes': ('smoke', 'fire'),
        'palette': [(0, 255, 255), (255, 0, 255)]
    }
    def __len__(self) -> int:
        return len(self.data_list)#//10