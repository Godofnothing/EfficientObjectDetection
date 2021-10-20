import numpy as np

from typing import List
from mmdet.core.evaluation import eval_map, eval_recalls


class DetectionReward:

    def __init__(
        self,
        lr_image_flops : int,
        hr_image_flops : int,
        flop_cost_weight: float,
        beta: float = 0.1,
        detection_metric : str = 'mAP',
        iou_thr: float = 0.5,
    ):
        self.lr_image_flops = lr_image_flops
        self.hr_image_flops = hr_image_flops
        self.flop_cost_weight = flop_cost_weight
        self.detection_metric  = detection_metric 
        self.iou_thr = iou_thr
        self.beta = beta
        
    def compute_reward(
        self, 
        lr_dt_results : List[List[np.ndarray]],
        hr_dt_results : List[List[np.ndarray]],
        lr_patches_bboxes : np.ndarray,
        hr_patches_bboxes : np.ndarray,
        labels : List[int],
        actions : List[int]
    ):
        '''
        Args:
            lr_dt_results : detection results for LR patches
            hr_dt_results : detection results for HR patches
            lr_patches_bboxes : bboxes for LR patches
            hr_patches_bboxes : bboxes for HR patches
            labels : list of categories for gt bboxes
            actions: list of actions, taken by the agent
        Returns:
            total_cost : the total cost of actions for each patch
        '''
        lr_dt_rewards, hr_dt_rewards = [], []
        # in case the detection metric is mAP
        if self.detection_metric  == 'mAP':
            for lr_dt_result, lr_patch_bboxes, patch_labels in zip(lr_dt_results, lr_patches_bboxes, labels):
                lr_image_annotations = [dict(bboxes=lr_patch_bboxes, labels=np.array(patch_labels))]
                lr_dt_rewards.append(eval_map([lr_dt_result], lr_image_annotations, logger=None, iou_thr=self.iou_thr)[0])

            for hr_dt_result, hr_patch_bboxes, patch_labels in zip(hr_dt_results, hr_patches_bboxes, labels):
                hr_image_annotations = [dict(bboxes=hr_patch_bboxes, labels=np.array(patch_labels))]
                hr_dt_rewards.append(eval_map([hr_dt_result], hr_image_annotations, logger=None, iou_thr=self.iou_thr)[0])
        # in case detection metric is recall
        if self.detection_metric  == 'recall':
            for lr_dt_result, lr_patch_bboxes in zip(lr_dt_results, lr_patches_bboxes):
                proposals = np.concatenate(lr_dt_result, axis=0)
                recalls = eval_recalls([lr_patch_bboxes], [proposals], proposal_nums=len(lr_patch_bboxes))[0][0]
                if np.isnan(recalls):
                    recalls = 0
                lr_dt_rewards.append(recalls)

            for hr_dt_result, hr_patch_bboxes in zip(hr_dt_results, hr_patches_bboxes):
                proposals = np.concatenate(hr_dt_result, axis=0)
                recalls = eval_recalls([hr_patch_bboxes], [proposals], proposal_nums=len(hr_patch_bboxes))[0][0]
                if np.isnan(recalls):
                    recalls = 0
                hr_dt_rewards.append(recalls)

        total_cost = 0.0
        for i in range(len(actions)):
            dt_reward = (actions[i] * hr_dt_rewards[i] + (1 - actions[i]) * (lr_dt_rewards[i] + self.beta))
            comp_reward = self.flop_cost_weight * (self.hr_image_flops - self.lr_image_flops) * (1 - actions[i])
            total_cost += (dt_reward + comp_reward)

        return total_cost
