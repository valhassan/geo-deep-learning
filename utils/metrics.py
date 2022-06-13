import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from sklearn.metrics import classification_report

import geopandas as gpd
from solaris.eval.base import Evaluator
from torchmetrics import MetricCollection, JaccardIndex

from dataset.aoi import AOI

min_val = 1e-6
def create_metrics_dict(num_classes):
    metrics_dict = {'precision': AverageMeter(), 'recall': AverageMeter(), 'fscore': AverageMeter(),
                    'loss': AverageMeter(), 'iou': AverageMeter()}

    for i in range(0, num_classes):
        metrics_dict['precision_' + str(i)] = AverageMeter()
        metrics_dict['recall_' + str(i)] = AverageMeter()
        metrics_dict['fscore_' + str(i)] = AverageMeter()
        metrics_dict['iou_' + str(i)] = AverageMeter()

    # Add overall non-background iou metric
    metrics_dict['iou_nonbg'] = AverageMeter()

    return metrics_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def report_classification(pred, label, batch_size, metrics_dict, ignore_index=-1):
    """Computes precision, recall and f-score for each class and average of all classes.
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    class_report = classification_report(label.cpu(), pred.cpu(), output_dict=True)

    class_score = {}
    for key, value in class_report.items():
        if key not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy'] and key != str(ignore_index):
            class_score[key] = value

            metrics_dict['precision_' + key].update(class_score[key]['precision'], batch_size)
            metrics_dict['recall_' + key].update(class_score[key]['recall'], batch_size)
            metrics_dict['fscore_' + key].update(class_score[key]['f1-score'], batch_size)

    metrics_dict['precision'].update(class_report['weighted avg']['precision'], batch_size)
    metrics_dict['recall'].update(class_report['weighted avg']['recall'], batch_size)
    metrics_dict['fscore'].update(class_report['weighted avg']['f1-score'], batch_size)

    return metrics_dict


def iou(pred, label, batch_size, num_classes, metric_dict, only_present=True):
    """Calculate the intersection over union class-wise and mean-iou"""
    ious = []
    pred = pred.cpu()
    label = label.cpu()
    for i in range(num_classes):
        c_label = label == i
        if only_present and c_label.sum() == 0:
            ious.append(np.nan)
            continue
        c_pred = pred == i
        intersection = (c_pred & c_label).float().sum()
        union = (c_pred | c_label).float().sum()
        iou = (intersection + min_val) / (union + min_val)  # minimum value added to avoid Zero division
        ious.append(iou)
        metric_dict['iou_' + str(i)].update(iou.item(), batch_size)

    # Add overall non-background iou metric
    c_label = (1 <= label) & (label <= num_classes - 1)
    c_pred = (1 <= pred) & (pred <= num_classes - 1)
    intersection = (c_pred & c_label).float().sum()
    union = (c_pred | c_label).float().sum()
    iou = (intersection + min_val) / (union + min_val)  # minimum value added to avoid Zero division
    metric_dict['iou_nonbg'].update(iou.item(), batch_size)

    mean_IOU = np.nanmean(ious)
    if (not only_present) or (not np.isnan(mean_IOU)):
        metric_dict['iou'].update(mean_IOU, batch_size)
    return metric_dict


def iou_torchmetrics(pred: torch.Tensor, label: torch.Tensor, device: str = 'cpu'):  # FIXME: merge with iou() above
    metrics = MetricCollection([JaccardIndex(num_classes=2, ignore_index=False, )], prefix="evaluate_").to(device)
    metrics(pred.to(device), label.to(device))
    results = metrics.compute()
    metrics.reset()
    iou = results["evaluate_JaccardIndex"].item()
    return iou


#### Benchmark Metrics ####
""" Segmentation Metrics from : https://github.com/jeremiahws/dlae/blob/master/metnet_seg_experiment_evaluator.py """

class ComputePixelMetrics():
    '''
    Compute pixel-based metrics between two segmentation masks.
    :param label: (numpy array) reference segmentaton mask
    :param pred: (numpy array) predicted segmentaton mask
    '''
    __slots__ = 'label', 'pred', 'num_classes'

    def __init__(self, label, pred, num_classes):
        self.label = label
        self.pred = pred
        self.num_classes = num_classes

    def update(self, metric_func):
        metric = {}
        classes = []
        for i in range(self.num_classes):
            c_label = self.label == i
            c_pred = self.pred == i
            m = metric_func(c_label, c_pred)
            classes.append(m)
            metric[metric_func.__name__ + '_' + str(i)] = classes[i]
        mean_m = np.nanmean(classes)
        metric['macro_avg_' + metric_func.__name__] = mean_m
        metric['micro_avg_' + metric_func.__name__] = metric_func(self.label, self.pred)
        return metric

    @staticmethod
    def iou(label, pred):
        '''
        :return: IOU
        '''
        intersection = (pred & label).sum()
        union = (pred | label).sum()
        iou = (intersection + min_val) / (union + min_val)

        return iou

    @staticmethod
    def dice(label, pred):
        '''
        :return: Dice similarity coefficient
        '''
        intersection = (pred & label).sum()
        dice = (2 * intersection) / ((label.sum()) + (pred.sum()))

        return dice


def iou_per_obj(
        pred: Union[str, Path],
        gt:Union[str, Path],
        attr_field: str = None,
        attr_vals: List = None,
        aoi_id: str = None,
        aoi_categ: str = None,
        min_iou: float = 0.5,
        gt_clip_bounds = None):
    """
    Calculate iou per object by comparing vector ground truth and vectorized prediction
    @param pred:
    @param gt:
    @param attr_field:
    @param attr_vals:
    @param aoi_id:
    @return:
    """
    if not aoi_id:
        aoi_id = Path(pred).stem
    # filter out non-buildings
    gt_gdf = gpd.read_file(gt, bbox=gt_clip_bounds)
    evaluator = Evaluator(ground_truth_vector_file=gt_gdf)

    evaluator.load_proposal(pred, conf_field_list=None)
    scoring_dict_list, TP_gdf, FN_gdf, FP_gdf = evaluator.eval_iou_return_GDFs(
        calculate_class_scores=False,
        miniou=min_iou
    )

    if TP_gdf is not None:
        TP_gdf.to_file(pred, layer='True_Pos', driver="GPKG")
    if FN_gdf is not None:
        FN_gdf.to_file(pred, layer='False_Neg', driver="GPKG")
    if FP_gdf is not None:
        FP_gdf.to_file(pred, layer='False_Pos', driver="GPKG")

    scoring_dict_list[0] = OrderedDict(scoring_dict_list[0])
    scoring_dict_list[0]['aoi'] = aoi_id
    scoring_dict_list[0].move_to_end('aoi', last=False)
    scoring_dict_list[0]['category'] = aoi_categ
    scoring_dict_list[0].move_to_end('category', last=False)
    logging.info(scoring_dict_list[0])
    return scoring_dict_list[0]
