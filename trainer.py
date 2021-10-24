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
from torch.utils.data import DataLoader
from utils.tools import StatManager, StatsSuiteManager 
from IPython.display import clear_output
from utils.metrics import DatasetMetricEvaluator
from utils.tools import stats_suite_manager_serialize


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
            val_sample = self.val_dataset[val_idx]
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

class TrainerSP:

    def __init__(
        self,
        agent,
        train_dataset,
        detection_reward : DetectionReward,
        val_dataset_metric_evaluator : DatasetMetricEvaluator,
        val_dataset = None,
        alpha = 0.8,
        device = 'cuda'):
        '''
        :Arguments:
        - agent: torch.nn.Module : agent to be trained
        - train_dataset : PolicyDataset
        - detection_reward : revard class
        - val_dataset : PolicyDataset
        - alpha : float : probability bounding factor (see original implementation)
        '''

        self.agent = agent
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_dataset.return_indices = True
        self.val_dataset_metric_evaluator = val_dataset_metric_evaluator
        self.detection_reward = detection_reward 
        self.device = device

        self.optimizer = None
        self.scheduler=None
        self.alpha = alpha
        self.verbose=0

    def configure_optimizer(self, optimizer):
        self.optimizer = optimizer

    def configure_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train_epoch(self, epoch, trainloader, ssm):
        assert self.optimizer is not None
        # set model to train mode
        self.agent.train()
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
            batch_size = inputs.size(0)
            # inputs are batch of images 
            # targets are metrics by detectors
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            assert targets.size(1) == 2

            # Actions by the Agent
            probs = F.sigmoid(self.agent(inputs)).view(-1)
            # it is sort of exploration
            alpha_hp = np.clip(self.alpha + epoch * 0.001, 0.6, 0.99)
            probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)

            # Sample the policies from the Bernoulli distribution characterized by agent
            distr = Bernoulli(probs)
            policy_sample = distr.sample()

            # Test time policy - used as baseline policy in the training step
            policy_map = torch.zeros_like(probs)
            policy_map[probs >= 0.5] = 1.0
            # print(policy_map.shape)
            # print(probs.shape)

            # compute the reward for the baseline and stoch models
            reward_baseline = self.detection_reward.compute_reward(targets[:, 0], targets[:, 1], policy_map)
            assert len(reward_baseline.shape) == 1
            assert reward_baseline.size(0) == batch_size

            reward_stoch = self.detection_reward.compute_reward(targets[:, 0], targets[:, 1], policy_sample)
            assert reward_baseline.shape == reward_stoch.shape

            advantage = reward_stoch - reward_baseline
            assert advantage.shape == policy_sample.shape

            # find the loss for the agent
            loss = - distr.log_prob(policy_sample)
            loss = loss * advantage
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ssm.add('train_RW_stoch', reward_stoch)
            ssm.add('train_RW_base', reward_baseline)
            ssm.add('train_policies_mean', policy_sample)
            ssm.add('train_policies_std', policy_sample)
            # ssm.add('loss', loss)
            if self.verbose:
                if batch_idx % self.verbose == 0:
                    clear_output(wait=True)
                    ssm.draw(0, 1, 2, 3, 4, 5, ncols=2, figsize=(15, 10))

    def val_epoch(self, epoch, valloader, ssm, _type='naive'):
        assert _type in ['naive', 'stoch']
        self.agent.eval()
        self.val_dataset_metric_evaluator.reset()
        for batch_idx, (inputs, targets, indices) in tqdm(enumerate(valloader), total=len(valloader)):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Actions by the Policy Network
            probs = F.sigmoid(self.agent(inputs)).view(-1)

            if _type == 'stoch':
                distr = Bernoulli(probs)
                policy = distr.sample()
            else:
                policy = torch.zeros_like(probs)
                policy[probs >= 0.5] = 1.0

            reward = self.detection_reward.compute_reward(targets[:, 0], targets[:, 1], policy)
            self.val_dataset_metric_evaluator.update(policy, indices)

            ssm.add('val_RW', reward)
            ssm.add('val_RW_mean', reward)
            # ssm.add('val_policies_mean', policy)
            # ssm.add('val_policies_std', policy)
            ssm.add('val_sample_metric', self.detection_reward.compute_sample_metric(
                targets[:, 0], targets[:, 1], policy))
            ssm.add('val_flops', self.detection_reward.compute_sample_flops(policy))
        dataset_metrics = self.val_dataset_metric_evaluator.evaluate('mAP')
        self.val_dataset_metric_evaluator.reset()
        ssm.add('val_dataset_mAP', dataset_metrics)
        # ssm.add('val_dataset_recall', dataset_metrics['recall'])
        clear_output(wait=True)
        ssm.draw(0, 1, 2, 3, 4, 5, ncols=2, figsize=(15, 10))


    def train(
        self, 
        num_epochs, 
        validate=False, 
        batch_size=64, 
        val_policy_type='naive',
        ssm_save_path = None,
        verbose=0):

        self.verbose=verbose
        ssm = StatsSuiteManager()
        ssm.register(StatManager('train_RW_stoch'), 0)
        ssm.register(StatManager('train_RW_base'), 0)
        ssm.register(StatManager('train_policies_mean'), 1)
        ssm.register(StatManager('train_policies_std', 'batch_std'), 1)

        if validate:
            ssm.register(StatManager('val_RW'), 2)
            ssm.register(StatManager('val_RW_mean', 'epoch_mean', False), 2)
            # ssm.register(StatManager('val_policies_mean', 'epoch_mean', False), 3)
            # ssm.register(StatManager('val_policies_std', 'epoch_std', False), 3)
            ssm.register(StatManager('val_dataset_mAP', 'epoch_mean', False), 5)
            # ssm.register(StatManager('val_dataset_recall', 'epoch_mean', False), 5)
            ssm.register(StatManager('val_sample_metric', 'epoch_mean', False), 4)
            ssm.register(StatManager('val_flops', 'epoch_mean', False), 3)

        for i_epoch in range(num_epochs):
            if validate and i_epoch == 0:
                valloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
                self.val_epoch(i_epoch, valloader, ssm, _type=val_policy_type)
                # update val history
                ssm.epoch(
                    'val_RW', 'val_RW_mean', 'val_dataset_mAP', 'val_sample_metric', 'val_flops')

            trainloader = DataLoader(
                    self.train_dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(i_epoch, trainloader, ssm)
            if self.scheduler:
                self.scheduler.step()

            if validate:
                valloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
                self.val_epoch(i_epoch, valloader, ssm, _type=val_policy_type)
                # update val history
            ssm.epoch()
            if ssm_save_path is not None:
                stats_suite_manager_serialize(ssm, ssm_save_path)

        return ssm