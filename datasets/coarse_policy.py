import cv2
import json
import numpy as np

from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset


class CoarsePolicyDataset(Dataset):

    def __init__(
        self, 
        data_dir : str,
        annotation_path: str,
        num_splits : int, 
        lr_patch_size: int,
        hr_patch_size: int,
        image_prefix: str='train',
        min_visibility: float = 0.3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_prefix = image_prefix
        self.lr_patch_size = (lr_patch_size,) * 2
        self.hr_patch_size = (hr_patch_size,) * 2
        self.num_splits = num_splits
        self.min_visibility = min_visibility

        with open(f"{data_dir}/annotations/{annotation_path}") as anno_file:
            annotations = json.load(anno_file)

        self.image_ids = []
        self.id2paths = {}
        self.id2bboxes = defaultdict(list)
        self.id2labels = defaultdict(list)

        # create mapping from dataset_id to image_id
        for image_data in annotations['images']:
            self.image_ids.append(image_data['id'])
            self.id2paths[image_data['id']] = image_data['file_name']

        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            self.id2bboxes[image_id].append(annotation['bbox'])
            self.id2labels[image_id].append(annotation['category_id'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = cv2.imread(f"{self.data_dir}/{self.image_prefix}/{self.id2paths[image_id]}")

        Ns = self.num_splits

        ps_y, ps_x = (image.shape[0] // Ns, image.shape[1] // Ns)
        image = image[:Ns * ps_y, :Ns * ps_x]
        patches = image.reshape((Ns, ps_y, Ns, ps_x, -1)).transpose((0, 2, 1, 3, 4)).reshape((Ns * Ns, ps_y, ps_x, -1))

        # load boxes and rescale to image_size
        bboxes = self.id2bboxes[image_id]
        labels = self.id2labels[image_id]

        # assign bboxes to patches
        bboxes_for_patches = [[] for _ in range(Ns * Ns)]
        labels_for_patches = [[] for _ in range(Ns * Ns)]
        for bbox, label in zip(bboxes, labels):
            x1, y1, w, h = bbox
            xc, yc = x1 + w / 2, y1 + h / 2
            x2, y2 = x1 + w, y1 + h
            # to what patch to be assigned
            pid_x, pid_y = int(xc // ps_x), int(yc // ps_y)
            pid = pid_y * Ns + pid_x
            # get coordinates inside patch
            xp1 = max(0, x1 - pid_x * ps_x)
            yp1 = max(0, y1 - pid_y * ps_y)
            xp2 = min(ps_x - 1, x2 - pid_x * ps_x)
            yp2 = min(ps_y - 1, y2 - pid_y * ps_y)
            # do not add this bounding box if only a part below specified threshold is left
            if (xp2 - xp1) * (yp2 - yp1) / (w * h) < self.min_visibility:
                continue

            bboxes_for_patches[pid].append((xp1, yp1, xp2, yp2))
            labels_for_patches[pid].append(label)

        for patch_idx, patch_bboxes in enumerate(bboxes_for_patches):
            if len(patch_bboxes) > 0:
                bboxes_for_patches[patch_idx] = np.stack(patch_bboxes, axis=0)
            else:
                bboxes_for_patches[patch_idx] = np.zeros((0, 4))
                labels_for_patches[patch_idx] = np.zeros((0,))

        # get coarse and fine patches
        lr_patches = [cv2.resize(patch, self.lr_patch_size) for patch in patches]
        hr_patches = [cv2.resize(patch, self.hr_patch_size) for patch in patches]

        lr_patch_bboxes = deepcopy(bboxes_for_patches)
        hr_patch_bboxes = deepcopy(bboxes_for_patches)
        # resize bboxes in lr_patches
        for patch_bboxes in lr_patch_bboxes:
            if len(patch_bboxes) > 0:
                patch_bboxes[:, [0, 2]] *= self.lr_patch_size[1] / ps_x
                patch_bboxes[:, [1, 3]] *= self.lr_patch_size[0] / ps_y
        # resize bboxes in hr_patches
        for patch_bboxes in hr_patch_bboxes:
            if len(patch_bboxes) > 0:
                patch_bboxes[:, [0, 2]] *= self.hr_patch_size[1] / ps_x
                patch_bboxes[:, [1, 3]] *= self.hr_patch_size[0] / ps_y

        return {
            "lr_patches" : lr_patches,
            "hr_patches" : hr_patches,
            "lr_patches_bboxes" : lr_patch_bboxes,
            "hr_patches_bboxes" : hr_patch_bboxes,
            "labels" : labels_for_patches
        }
