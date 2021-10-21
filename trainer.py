import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions import Bernoulli
from mmdet.apis import inference_detector

from utils import DetectionReward, convert_predictions
from visualization import DetectionVisualizer


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


    def train_epoch(self):
        # set model to train mode
        self.agent.train()
        rewards_stoch, rewards_baseline = [], []
        policies = []
        # get random indices
        train_indices = np.random.permutation(len(self.train_dataset))
        for train_idx in tqdm(train_indices, total=len(train_indices)):
            train_sample = train_sample[train_idx]
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
            policy_baseline[probs >= 0.5] = 1
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

    
    def val_epoch(self):
         # set model to eval mode
        self.agent.eval()
        rewards_stoch = []
        policies = []
        # get random indices
        val_indices = np.random.permutation(len(self.val_dataset))
        for val_idx in tqdm(val_indices, total=len(val_indices)):
            val_sample = val_sample[val_idx]
            # get data from sample
            lr_patches = val_sample['lr_patches']
            hr_patches = val_sample['hr_patches']
            lr_patches_bboxes = val_sample['lr_patches_bboxes']
            hr_patches_bboxes = val_sample['hr_patches_bboxes']
            labels = val_sample['labels']

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

            # Find the reward for stochastic policy
            reward_stoch = self.detection_reward.compute_reward(
                lr_dt_results, hr_dt_results, lr_patches_bboxes, hr_patches_bboxes, labels, policy_stoch
            )

            rewards_stoch.append(reward_stoch.cpu())
            policies.append(policy_stoch.data.cpu())

        return rewards_stoch, policies


    def train(self, num_epochs, validate=False):
        train_history = dict(
            rewards_stoch=[],
            rewards_baseline=[],
            policies=[]
        )

        if validate:
            val_history = dict(
                rewards_stoch=[],
                policies=[]
            )

        for _ in range(num_epochs):
            rewards_stoch, rewards_baseline, policies = self.train_epoch()
            if self.scheduler:
                self.scheduler.step()
            # update train history
            train_history['rewards_stoch'].append(get_mean(rewards_stoch))
            train_history['rewards_baseline'].append(get_mean(rewards_baseline))
            train_history['policies'].append(get_mean(policies))

            if validate:
                rewards_stoch, policies = self.val_epoch()
                # update val history
                val_history['rewards_stoch'].append(get_mean(rewards_stoch))
                val_history['policies'].append(get_mean(policies))

        if validate:
            return train_history, val_history
        else:
            return train_history


    def configure_optimizers(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler


    def configure_visualizer(self, visualizer : DetectionVisualizer):
        self.visualizer = visualizer


    def visualize_random_prediction(self, dataset='train', output_path=''):
        if dataset == 'train':
            idx = np.random.randint(0, len(self.train_dataset))
            sample = self.train_dataset[idx]
        elif dataset == 'val':
            idx = np.random.randint(0, len(self.val_dataset))
            sample = self.val_dataset[idx]

        lr_patches = sample['lr_patches']
        hr_patches = sample['hr_patches']
        lr_patches_bboxes = sample['lr_patches_bboxes']
        hr_patches_bboxes = sample['hr_patches_bboxes']
        labels = sample['labels']

        # create tensors and move to device (TODO) add normalization
        lr_batch = torch.from_numpy(np.stack(lr_patches, axis=0))
        hr_batch = torch.from_numpy(np.stack(hr_patches, axis=0))

        # compute probs
        probs = self.agent(lr_batch)
         # stochastic policy
        distr = Bernoulli(probs)
        policy_stoch = distr.sample()

        chosen_patches = [
            hr_patches[i] if policy_stoch[i][0] else lr_patches[i] for i in range(len(labels))
        ]

        # get detection results
        dt_results = inference_detector(self.detector, chosen_patches)
        # convert predictions to suitable format
        dt_bboxes, dt_labels = convert_predictions(dt_results)
        # draw predictions
        self.visualizer.draw_patches_with_bboxes(chosen_patches, dt_bboxes, dt_labels, output_path)        
