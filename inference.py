import cv2
import numpy as np
import torch.nn as nn

from torchvision import transforms
from torch.distributions import Bernoulli
from mmdet.apis import inference_detector

from datasets import ImageFolder


DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        [123.675/255, 116.28/255, 103.53/255], 
        [58.395/255, 57.12/255, 57.375/255]
    )
])


class Inference:

    def __init__(
        self, 
        agent  : nn.Module, 
        wk_detector : nn.Module,
        sg_detector : nn.Module,
        train_dataset: ImageFolder = None,
        val_dataset: ImageFolder = None,
        preproc_transforms = DEFAULT_TRANSFORMS
    ):
        self.agent = agent
        self.wk_detector = wk_detector
        self.sg_detector = sg_detector
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.preproc_transforms = preproc_transforms
 
    def inference_by_idx(self, idx: int, dataset='train'):
        if dataset == 'train':
            assert idx < len(self.train_dataset)
            raw_image = self.train_dataset.get_raw_image(idx)
            processed_image = self.train_dataset[idx]

        if dataset == 'val':
            assert idx < len(self.val_dataset)
            raw_image = self.val_dataset.get_raw_image(idx)
            processed_image = self.val_dataset[idx]
        
        probs = self.agent(processed_image.unsqueeze(0))
        distr = Bernoulli(probs)
        action = distr.sample()

        detector = self.sg_detector if action else self.wk_detector
        dt_results = inference_detector(detector, raw_image)

        return raw_image, action, dt_results


    def inference_random(self, dataset='train'):
        if dataset == 'train':
            idx = np.random.randint(0, len(self.train_dataset))
        if dataset == 'val':
            idx = np.random.randint(0, len(self.val_dataset))
        return self.inference_by_idx(idx, dataset)


    def inference_by_path(self, path_to_image: str):
        assert self.preproc_transforms, "you need to speciy preprocessing pipeline for inferencing images"

        raw_image = cv2.imread(path_to_image)
        processed_image = self.preproc_transforms(raw_image)

        probs = self.agent(processed_image.unsqueeze(0))
        distr = Bernoulli(probs)
        action = distr.sample()

        detector = self.sg_detector if action else self.wk_detector
        dt_results = inference_detector(detector, raw_image)

        return raw_image, action, dt_results
