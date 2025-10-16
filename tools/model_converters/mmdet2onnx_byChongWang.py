# Copyright (c) iFireTEK (Chong Wang). All rights reserved.
import torch
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
import copy
import os.path as osp
import warnings
import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmengine.visualization import Visualizer
from rich.progress import track

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import encode_mask_results, mask2bbox
from mmdet.utils import ConfigType
from mmdet.evaluation import get_classes

try:
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None
    VOID = None

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    call_args = vars(parser.parse_args())


    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

class Det2ONNX(DetInferencer):
    def __init__(self,init_args,call_args):
        super().__init__(**init_args)
        self.out_dir = call_args['out_dir']

    def pre_process_wrap(self,
                         inputs: InputsType,
                         batch_size: int = 1,
                         return_vis: bool = False,
                         show: bool = False,
                         wait_time: int = 0,
                         no_save_vis: bool = False,
                         draw_pred: bool = True,
                         pred_score_thr: float = 0.3,
                         return_datasamples: bool = False,
                         print_result: bool = False,
                         no_save_pred: bool = True,
                         out_dir: str = '',
                         # by open image task
                         texts: Optional[Union[str, list]] = None,
                         # by open panoptic task
                         stuff_texts: Optional[Union[str, list]] = None,
                         # by GLIP
                         custom_entities: bool = False,
                         **kwargs) -> dict:
        (
            self.preprocess_kwargs,
            self.forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        if texts is not None and isinstance(texts, str):
            texts = [texts] * len(ori_inputs)
        if stuff_texts is not None and isinstance(stuff_texts, str):
            stuff_texts = [stuff_texts] * len(ori_inputs)
        if texts is not None:
            assert len(texts) == len(ori_inputs)
            for i in range(len(texts)):
                if isinstance(ori_inputs[i], str):
                    ori_inputs[i] = {
                        'text': texts[i],
                        'img_path': ori_inputs[i],
                        'custom_entities': custom_entities
                    }
                else:
                    ori_inputs[i] = {
                        'text': texts[i],
                        'img': ori_inputs[i],
                        'custom_entities': custom_entities
                    }
        if stuff_texts is not None:
            assert len(stuff_texts) == len(ori_inputs)
            for i in range(len(stuff_texts)):
                ori_inputs[i]['stuff_text'] = stuff_texts[i]

        inputs = self.preprocess(ori_inputs, batch_size=batch_size, **self.preprocess_kwargs)
        return inputs


    def export(self,data):
        # preds = self.model.test_step(data)
        output_names = ['dets', 'labels']
        input_name = 'input'
        # data.pop('data_samples')
        data['inputs']= data['inputs'][0].unsqueeze(0).float().cuda()
        torch.onnx.export(
            self.model,
            data,
            self.out_dir,
            input_names=[input_name],
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True)
        pass

def main():
    init_args, call_args = parse_args()
    inferencer = Det2ONNX(init_args,call_args)
    inputs = inferencer.pre_process_wrap(**call_args)
    for ori_imgs, data in track(inputs, description='Inference') :
        preds = inferencer.export(data)
        pass


    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()