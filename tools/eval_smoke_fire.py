import copy,json,argparse
import numpy as np
from collections import OrderedDict
from mmengine.logging import print_log
from mmdet.evaluation.functional.mean_ap import eval_map
import os.path as osp
import mmcv,cv2,scipy,os,mmengine
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description='fire smoke evaluator')
    parser.add_argument('path_res', help='results path')
    parser.add_argument('path_ann', help='xml files path')
    parser.add_argument('path_save', help='eval json saved path')
    parser.add_argument('--iou_thr',  type=float,        default=0.1,)
    parser.add_argument('--score_thr',  type=float,        default=0.1,)
    parser.add_argument('--metrics',  type=list,        default=['mAP', 'bbox', 'fireACC_img', 'smokeACC_img'])
    args = parser.parse_args()
    return args


class DetectionRectsEval:
    def __init__(self,iou_threshod=0.1):
        self.tp=0
        self.fp=0
        self.fn=0
        self.iou_threshod = iou_threshod
        self.pred_matched_all = []

    def cal_iou_mat (self,rects1,rects2):
        iou_mat =np.zeros((len(rects1),len(rects2)),dtype=float)
        bboxes1 = np.array(rects1)
        bboxes2 = np.array(rects2)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] ) * ( bboxes1[:, 3] - bboxes1[:, 1] )
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] ) * (  bboxes2[:, 3] - bboxes2[:, 1] )
        for i in range(bboxes1.shape[0]):
            x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
            y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
            x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
            y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
            overlap = np.maximum(x_end - x_start , 0) * np.maximum( y_end - y_start , 0)
            union = area1[i] + area2 - overlap
            union = np.maximum(union, 1e-10)
            iou_mat[i, :] = overlap / union
        return iou_mat

    def updata(self,gt_rects,pred_rects):
        '''rects的格式为[x_min,y_min,x_max,y_max]'''
        if len(gt_rects)+len(pred_rects)==0:
            self.gt_matched = np.array([],dtype=bool)
            self.pred_matched = np.array([],dtype=bool)
            return
        if len(gt_rects)==0:
            self.fp += len(pred_rects)
            self.gt_matched = np.array([],dtype=bool)
            self.pred_matched = np.array([False]*len(pred_rects),dtype=bool)
            return
        if len(pred_rects)==0:
            self.fn +=len(gt_rects)
            self.gt_matched = np.array([False]*len(gt_rects),dtype=bool)
            self.pred_matched = np.array([],dtype=bool)
            return
        iou_mat = self.cal_iou_mat(gt_rects,pred_rects)
        match_index_list = scipy.optimize.linear_sum_assignment(cost_matrix=1-iou_mat)
        matched_mat = np.zeros_like(iou_mat)
        matched_mat[match_index_list[0],match_index_list[1]] =1
        iou_mat_matched = iou_mat * matched_mat
        iou_mat_matched_T = iou_mat_matched>=self.iou_threshod
        self.gt_matched = np.max(iou_mat_matched_T,axis=1)
        self.pred_matched = np.max(iou_mat_matched_T,axis=0)
        self.pred_matched_all.append(copy.deepcopy(self.pred_matched))
        self.tp +=np.sum(self.pred_matched*1)
        self.fp +=len(pred_rects) - np.sum(self.pred_matched*1)
        self.fn +=len(gt_rects) - np.sum(self.pred_matched*1)


    def get(self):
        self.recall = self.tp/(1e-10+self.tp+self.fn)
        self.precision = self.tp/(1e-10+self.tp+self.fp)
        self.f1 = self.recall*self.precision*2/(self.recall+self.precision + 1e-10)


class smoke_fire_evaluator:
    def __init__(self,
                 path_res,
                 path_ann,
                 iou_thr=0.1,
                 score_thr=0.1,
                 metrics=['mAP', 'bbox', 'fireACC_img', 'smokeACC_img']):
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.CLASSES=['fire','smoke']
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.metrics = metrics
        res_lst = os.listdir(path_res)
        res_lst = [v for v in res_lst if v.endswith('.json')]
        ann_lst = os.listdir(path_ann)
        ann_lst = [v for v in ann_lst if v.endswith('.xml')]
        ann_lst.sort()
        res_lst_new=[]
        for ann in ann_lst:
            if ann.replace('.xml','.json') in res_lst:
                res_lst_new.append( ann.replace('.xml','.json'))
            else:
                print(ann + '  has no correspondding prediction json file')
        self.ann_lst = [os.path.join(path_ann,v) for v in ann_lst]
        self.res_lst = [os.path.join(path_res, v) for v in res_lst_new]
        self.video_dict = self.obtain_video_dict_accordname()

    def obtain_video_dict_accordname(self):
        video_dict={}
        for i,name in enumerate(self.ann_lst):
            name = '--'.join(os.path.split(name)[-1].split('--')[:-1])
            if name not in video_dict.keys():
                video_dict[name] = [i]
            else:
                video_dict[name].append(i)
        return video_dict


    def get_ann_info(self, xml_path):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        if osp.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.CLASSES:
                    continue
                label = self.cat2label[name]
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
                bnd_box = obj.find('bndbox')
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                ignore = False
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_res_info(self,json_path):
        with open(json_path,'r') as f:
            res = json.load(f)
        pred = [[],[]]
        for label,score,bbox in zip(res['labels'], res['scores'], res['bboxes']):
            pred[1-label].append(bbox+[score])  #模型预测的是烟在前火在后，这里把结果调整一下
        pred =[np.array(v,dtype=np.float32) if len(v)>0 else np.zeros(shape=(0,5),dtype=np.float32) for v in pred ]

        return pred

    def evaluate(self,logger=None):
        allowed_metrics = ['mAP', 'bbox', 'fireACC_img', 'smokeACC_img']
        metric_list = self.metrics
        results = [self.get_res_info(json_path=i) for i in self.res_lst]
        annotations = [self.get_ann_info(xml_path=i) for i in self.ann_lst[:182]]
        assert len(results)==len(annotations)
        label_fire = np.array([np.sum(ann['labels'] == 0) for ann in annotations], dtype=int)
        label_fire = np.array((label_fire >= 1) * 1, dtype=int)
        label_smoke = np.array([np.sum(ann['labels'] == 1) for ann in annotations], dtype=int)
        label_smoke = np.array((label_smoke >= 1) * 1, dtype=int)

        iou_thrs = [self.iou_thr] if isinstance(self.iou_thr, float) else self.iou_thr
        eval_results = OrderedDict()
        eval_results['score_thr'] = self.score_thr

        for metric in metric_list:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
            if metric == 'mAP':
                assert isinstance(iou_thrs, list)
                ds_name = self.CLASSES
                mean_aps = []
                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, ap_res = eval_map(
                        results,
                        annotations,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset=ds_name,
                        logger=logger,
                        use_legacy_coordinate=True)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
                eval_results.move_to_end('mAP', last=False)
                eval_results['AP'] = ap_res

            elif metric == 'bbox':
                eval_results['bbox'] = {}
                for score_thr in points:
                    current_result = {}
                    evaluator_fire = DetectionRectsEval(iou_threshod=0.1)
                    evaluator_smoke = DetectionRectsEval(iou_threshod=0.1)
                    for ann, pred in zip(annotations, results):
                        ann_fire = list(ann['bboxes'][ann['labels'] == 0])
                        pred_fire = list(pred[0][pred[0][:, -1] > score_thr][:, 0:-1])
                        evaluator_fire.updata(gt_rects=ann_fire, pred_rects=pred_fire)

                        ann_smoke = list(ann['bboxes'][ann['labels'] == 1])
                        pred_smoke = list(pred[1][pred[1][:, -1] > score_thr])
                        evaluator_smoke.updata(gt_rects=ann_smoke, pred_rects=pred_smoke)

                    evaluator_fire.get()
                    evaluator_smoke.get()
                    print('score_thr:%.4f  fire: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
                          % (score_thr, evaluator_fire.tp, evaluator_fire.fp, evaluator_fire.fn, evaluator_fire.recall,
                             evaluator_fire.precision, evaluator_fire.f1))
                    print('score_thr:%.4f   smoke: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
                          % (score_thr, evaluator_smoke.tp, evaluator_smoke.fp, evaluator_smoke.fn, evaluator_smoke.recall,
                             evaluator_smoke.precision, evaluator_smoke.f1))

                    current_result['fire_TP'] = evaluator_fire.tp
                    current_result['fire_FP'] = evaluator_fire.fp
                    current_result['fire_FN'] = evaluator_fire.fn
                    current_result['fire_recall'] = evaluator_fire.recall,
                    current_result['fire_precision'] = evaluator_fire.precision
                    current_result['fire_f1'] = evaluator_fire.f1
                    current_result['smoke_TP'] = evaluator_smoke.tp
                    current_result['smoke_FP'] = evaluator_smoke.fp
                    current_result['smoke_FN'] = evaluator_smoke.fn
                    current_result['smoke_recall'] = evaluator_smoke.recall,
                    current_result['smoke_precision'] = evaluator_smoke.precision
                    current_result['smoke_f1'] = evaluator_smoke.f1
                    eval_results['bbox']['%f' % score_thr] = current_result
            elif metric == 'fireACC_img':
                gt_img = label_fire
                eval_results['img_fire'] = {}
                for score_thr in points:
                    current_result = {}
                    pred_img = []
                    score_img = []
                    for img_i, pred in enumerate(results):
                        pred = pred[0]
                        if pred.shape[0] == 0:
                            pred_img.append(0)
                            score_img.append(0.0)
                            continue
                        else:
                            score_img.append(np.max(pred[:, -1]))
                        whithout_pos = np.alltrue(pred[:, -1] < score_thr)
                        if whithout_pos:
                            pred_img.append(0)
                        else:
                            pred_img.append(1)
                    pred_img = np.array(pred_img, dtype=int)
                    score_img = np.array(score_img)
                    eval_results['img_fire_score'] = copy.deepcopy(score_img)
                    eval_results['img_fire_gt'] = copy.deepcopy(gt_img)
                    current_result['img_fire_TP'] = np.sum(pred_img * gt_img)
                    current_result['img_fire_FP'] = np.sum(pred_img * (1 - gt_img))
                    current_result['img_fire_FN'] = np.sum((1 - pred_img) * gt_img)
                    current_result['img_fire_TN'] = np.sum((1 - pred_img) * (1 - gt_img))
                    current_result['img_fire_ACC'] = np.sum(pred_img == gt_img) / gt_img.shape[0]
                    current_result['img_fire_precision'] = current_result['img_fire_TP'] / (
                                current_result['img_fire_TP'] + current_result['img_fire_FP'] + 1e-10)
                    current_result['img_fire_recall'] = current_result['img_fire_TP'] / (
                                current_result['img_fire_TP'] + current_result['img_fire_FN'] + 1e-10)
                    current_result['img_fire_f1'] = current_result['img_fire_recall'] * current_result[
                        'img_fire_precision'] * 2 / (current_result['img_fire_recall'] + current_result[
                        'img_fire_precision'] + 1e-10)

                    pred_video, gt_video, score_video = [], [], []
                    for video in self.video_dict.keys():
                        ids = np.array(self.video_dict[video])
                        pred = np.max(pred_img[ids])
                        gt = np.max(gt_img[ids])
                        pred_video.append(pred)
                        gt_video.append(gt)
                        score = np.max(score_img[ids])
                        score_video.append(score)
                    pred_video, gt_video = np.array(pred_video, dtype=int), np.array(gt_video, dtype=int)

                    eval_results['vider_fire_score'] = np.array(score_video)
                    eval_results['vider_fire_gt'] = np.array(gt_video)
                    current_result['video_fire_TP'] = np.sum(pred_video * gt_video)
                    current_result['video_fire_FP'] = np.sum(pred_video * (1 - gt_video))
                    current_result['video_fire_FN'] = np.sum((1 - pred_video) * gt_video)
                    current_result['video_fire_TN'] = np.sum((1 - pred_video) * (1 - gt_video))
                    current_result['video_fire_ACC'] = np.sum(pred_video == gt_video) / gt_video.shape[0]
                    current_result['video_fire_precision'] = current_result['video_fire_TP'] / (
                                current_result['video_fire_TP'] + current_result['video_fire_FP'] + 1e-10)
                    current_result['video_fire_recall'] = current_result['video_fire_TP'] / (
                                current_result['video_fire_TP'] + current_result['video_fire_FN'] + 1e-10)
                    current_result['video_fire_f1'] = current_result['video_fire_recall'] * current_result[
                        'video_fire_precision'] * 2 / \
                                                      (current_result['video_fire_recall'] + current_result[
                                                          'video_fire_precision'] + 1e-10)
                    eval_results['img_fire']['%f' % score_thr] = current_result
            elif metric == 'smokeACC_img':
                gt_img = label_smoke
                eval_results['img_smoke'] = {}
                for score_thr in points:
                    pred_img = []
                    score_img = []
                    current_result = {}
                    for img_i, pred in enumerate(results):
                        pred = pred[1]
                        if pred.shape[0] == 0:
                            pred_img.append(0)
                            score_img.append(0.0)
                            continue
                        else:
                            score_img.append(np.max(pred[:, -1]))
                        whithout_pos = np.alltrue(pred[:, -1] < score_thr)
                        if whithout_pos:
                            pred_img.append(0)
                        else:
                            pred_img.append(1)
                    pred_img = np.array(pred_img, dtype=int)
                    score_img = np.array(score_img, dtype=float)
                    eval_results['img_smoke_score'] = score_img
                    eval_results['img_smoke_gt'] = gt_img
                    current_result['img_smoke_TP'] = np.sum(pred_img * gt_img)
                    current_result['img_smoke_FP'] = np.sum(pred_img * (1 - gt_img))
                    current_result['img_smoke_FN'] = np.sum((1 - pred_img) * gt_img)
                    current_result['img_smoke_TN'] = np.sum((1 - pred_img) * (1 - gt_img))
                    current_result['img_smoke_ACC'] = np.sum(pred_img == gt_img) / gt_img.shape[0]
                    current_result['img_smoke_precision'] = current_result['img_smoke_TP'] / (
                                current_result['img_smoke_TP'] + current_result['img_smoke_FP'] + 1e-10)
                    current_result['img_smoke_recall'] = current_result['img_smoke_TP'] / (
                                current_result['img_smoke_TP'] + current_result['img_smoke_FN'] + 1e-10)
                    current_result['img_smoke_f1'] = current_result['img_smoke_recall'] * current_result[
                        'img_smoke_precision'] * 2 / (current_result['img_smoke_recall'] + current_result[
                        'img_smoke_precision'] + 1e-10)

                    pred_video, gt_video, score_video = [], [], []
                    for video in self.video_dict.keys():
                        ids = np.array(self.video_dict[video])
                        pred = np.max(pred_img[ids])
                        gt = np.max(gt_img[ids])
                        pred_video.append(pred)
                        gt_video.append(gt)
                        score_video.append(np.max(score_img[ids]))

                    pred_video, gt_video = np.array(pred_video, dtype=int), np.array(gt_video, dtype=int)
                    score_video = np.array(score_video)

                    eval_results['video_smoke_score'] = score_video
                    eval_results['video_smoke_gt'] = gt_video
                    current_result['video_smoke_TP'] = np.sum(pred_video * gt_video)
                    current_result['video_smoke_FP'] = np.sum(pred_video * (1 - gt_video))
                    current_result['video_smoke_FN'] = np.sum((1 - pred_video) * gt_video)
                    current_result['video_smoke_TN'] = np.sum((1 - pred_video) * (1 - gt_video))
                    current_result['video_smoke_ACC'] = np.sum(pred_video == gt_video) / gt_video.shape[0]
                    current_result['video_smoke_precision'] = current_result['video_smoke_TP'] / (
                                current_result['video_smoke_TP'] + current_result['video_smoke_FP'] + 1e-10)
                    current_result['video_smoke_recall'] = current_result['video_smoke_TP'] / (
                                current_result['video_smoke_TP'] + current_result['video_smoke_FN'] + 1e-10)
                    current_result['video_smoke_f1'] = current_result['video_smoke_recall'] * current_result[
                        'video_smoke_precision'] * 2 / \
                                                       (current_result['video_smoke_recall'] + current_result[
                                                           'video_smoke_precision'] + 1e-10)
                    eval_results['img_smoke']['%f' % score_thr] = current_result
        return eval_results


if __name__=="__main__":
    points = list(np.linspace(0, 0.05, 500, endpoint=False)) + list( np.linspace(0.05, 0.95, 500, endpoint=False)) + list(np.linspace(0.95, 1, 500, endpoint=False))
    # points = list(np.linspace(0, 0.05, 10, endpoint=False)) + list( np.linspace(0.05, 0.95, 10, endpoint=False)) + list(np.linspace(0.95, 1, 10, endpoint=False))

    args = parse_args()
    evaluator = smoke_fire_evaluator(path_res=args.path_res,path_ann=args.path_ann, iou_thr=args.iou_thr,score_thr=args.score_thr,metrics=args.metrics)
    eval_results = evaluator.evaluate()
    eval_results = {'metric':eval_results}
    mmengine.dump(eval_results, args.path_save)
