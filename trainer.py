import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions import Bernoulli
from mmdet.apis import inference_detector


from utils import DetectionReward


def get_mean(x : list):
    return sum(x) / len(x)


class Trainer:

    def __init__(
        self, 
        agent, 
        detector,
        train_dataset,
        detection_reward : DetectionReward,
        val_dataset = None,
        device = 'cuda'
    ):
        self.agent = agent
        self.detector = detector
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.detection_reward = detection_reward 
        self.device = device

        self.optimizer = None
        self.scheduler = None

    def train_epoch(self, epoch):
        rewards_stoch, rewards_baseline = [], []
        policies = []
        # get random indices
        train_indices = np.random.permutation(len(self.train_dataset))
        for train_idx in tqdm(train_indices, total=len(train_indices)):
            train_sample = self.train_dataset[train_idx]
            # get data from sample
            lr_patches = train_sample['lr_patches']
            hr_patches = train_sample['hr_patches']
            lr_patches_bboxes = train_sample['lr_patches_bboxes']
            hr_patches_bboxes = train_sample['hr_patches_bboxes']
            labels = train_sample['labels']

            # get detection results
            lr_dt_results = inference_detector(self.detector, lr_patches)
            hr_dt_results = inference_detector(self.detector, hr_patches)

            # create tensors and move to device (TODO) add normalization
            lr_batch = torch.from_numpy(np.stack(lr_patches, axis=0))
            hr_batch = torch.from_numpy(np.stack(hr_patches, axis=0))

            # Actions by the Agent (TODO)
            # probs = F.sigmoid(agent.forward(inputs))
            # alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
            # probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)
            probs = None

            # stochastic policy
            distr = Bernoulli(probs)
            policy_stoch = distr.sample()
            # deterministic policy
            policy_baseline = torch.zeros_like(probs, dtype=torch.int)
            policy_baseline[policy_baseline >= 0.5] = 1
            policy_baseline = Variable(policy_baseline)

            # Find the reward for deterministic and stochastic policy
            reward_baseline = self.detection_reward.compute_reward(
                lr_dt_results, hr_dt_results, lr_patches_bboxes, hr_patches_bboxes, labels, policy_baseline
            )
            reward_stoch = self.detection_reward.compute_reward(
                lr_dt_results, hr_dt_results, lr_patches_bboxes, hr_patches_bboxes, labels, policy_stoch
            )

            advantage = reward_stoch.to(self.device).float() - reward_baseline.to(self.device).float()

            # find the loss for the agent
            loss = -distr.log_prob(policy_stoch)
            loss = loss * Variable(advantage).expand_as(policy_stoch)
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            rewards_stoch.append(reward_stoch.cpu())
            rewards_baseline.append(reward_baseline.cpu())
            policies.append(policy_stoch.data.cpu())

        return rewards_stoch, rewards_baseline, policies


    def train(self, num_epochs):
        train_history = dict(
            rewards_stoch=[],
            rewards_baseline=[],
            policies=[]
        )

        for _ in range(num_epochs):
            rewards_stoch, rewards_baseline, policies = self.train_epoch()
            if self.scheduler:
                self.scheduler.step()

            train_history['rewards_stoch'].append(get_mean(rewards_stoch))
            train_history['rewards_baseline'].append(get_mean(rewards_baseline))
            train_history['policies'].append(get_mean(policies))

        return train_history


    def configure_optimizers(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
