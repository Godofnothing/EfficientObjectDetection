import torch
import numpy as np
import pickle
from mmdet.core.evaluation import eval_map, eval_recalls

class DatasetMetricEvaluator:

    @staticmethod
    def _process_detections(detection_path):
        with open(detection_path, 'rb') as fp:
            data = pickle.load(fp)
        res = [data[i][0] for i in range(len(data))]
        return res

    @staticmethod
    def _process_gt_detections(detection_path):
        with open(detection_path, 'rb') as fp:
            data = pickle.load(fp)
        res = {id: dict(bboxes=val[0], labels=val[1]) for id, val in data.items()}
        # res = [dict(bboxes=data[i][0], labels=data[i][1]) for i in range(len(data))]
        return res

    def __init__(self, dataset, wk_detection_path, sg_detection_path, gt_detection_path):
        '''
        :Parameters:
            metric : str : 'mAP' or 'recall' - metric to 
        evaluate on the dataset
            dataset: PolicyDataset : dataset to evaluate
        wk_detection_path: path to weak model detection results
        sg_detection_path: path to strong model detection results
        gt_detection_path: path to ground truth detection results
        '''
        self.dataset = dataset
        self.wk_detection = self._process_detections(wk_detection_path)
        self.sg_detection = self._process_detections(sg_detection_path)
        self.gt_detection = self._process_gt_detections(gt_detection_path)
        self.policies = []
        self.dataset_indices = []

    def reset(self):
        self.policies = []
        self.dataset_indices = []

    def update(self, policy, data_indices):
        policy = policy.detach().cpu().numpy().astype(np.int32).tolist()
        data_indices = data_indices.detach().cpu().numpy().astype(np.int32).tolist()
        self.policies.extend(policy)
        self.dataset_indices.extend(data_indices)

    def evaluate(self, metric, iou_thr=0.5):
        assert metric in ['mAP', 'recall', 'both']
        resulting_dt = []
        resulting_gt_dt = []
        for i in range(len(self.dataset_indices)):
            data_idx = self.dataset.number_in_annotations(
                self.dataset_indices[i])
            image_id = self.dataset.image_ids[
                self.dataset_indices[i]]
            # assert data_idx == self.dataset_indices[i]
            # assert self.dataset_indices[i] == i
            policy = self.policies[i]
            if policy == 0:
                curr_dt = self.wk_detection[data_idx]
            elif policy == 1:
                curr_dt = self.sg_detection[data_idx] 
            resulting_dt.append(curr_dt)
            resulting_gt_dt.append(self.gt_detection[image_id])
        if metric in ['mAP', 'both']:
            print('start mAP estimation')
            mAP = eval_map(resulting_dt, resulting_gt_dt, iou_thr=iou_thr, logger='silent')[0]
            print('finish mAP estimation')
            if metric == 'mAP':
                return mAP
        if metric in ['recall', 'both']:
            num_ims = len(resulting_dt)
            proposals = [np.concatenate(resulting_dt[i], axis=0) for i in range(num_ims)]
            res_gt = [resulting_gt_dt[i]['bboxes'] for i in range(num_ims)]
            proposal_nums = [len(res_gt[i]) for i in range(len(res_gt))]
            print('start recall estimation')
            recall = eval_recalls(
                res_gt, proposals, proposal_nums, logger='silent', iou_thrs=iou_thr)[0][0]
            print('finish recall estimation')
            if metric == 'recall':
                return recall
        return {'mAP': mAP, 'recall': recall}
