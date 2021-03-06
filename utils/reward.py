import torch

class DetectionReward:

    def __init__(
        self,
        wk_det_flops : int,
        sg_det_flops : int,
        flops_weight: float
    ):
        self.wk_det_flops = wk_det_flops
        self.sg_det_flops = sg_det_flops
        self.flops_weight = flops_weight

    def compute_reward(
        self, 
        wk_detection_metric ,
        sg_detection_metric ,
        action
    ):
        '''
        Args:
            wk_detection_metrics : detection metrics (mAP or recall) for weaker model
            sg_detection_metrics : detection metrics (mAP or recall) for stronger model
            actions: list of actions, taken by the agent
        Returns:
            total_cost : the total cost of actions for each patch
        '''
        dt_reward = action * sg_detection_metric + (1 - action) * wk_detection_metric
        comp_reward = self.flops_weight * (self.sg_det_flops - self.wk_det_flops) * (1 - action)

        return dt_reward + comp_reward

    def compute_sample_metric(
        self, 
        wk_detection_metric,
        sg_detection_metric,
        action):
        '''
        Args:
            see `compute_reward` function
        Returns:
            returns the sample metric given the action
        '''
        dt_metric = action * sg_detection_metric + (1 - action) * wk_detection_metric
        return dt_metric

    def compute_sample_flops(
        self,
        action, mul_by_flops_weight=True):
        '''
        Args:
            actions: the action, taken by the agent
        Returns:
            Elapsed flops to perform the detection
        '''
        dt_metric = action * self.sg_det_flops + (1 - action) * self.wk_det_flops
        if mul_by_flops_weight:
            dt_metric *= self.flops_weight
        return dt_metric

