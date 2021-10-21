import cv2
import json
import numpy as np

from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import pandas as pd
import torchvision.transforms as transforms

class PolicyDataset(Dataset):

    @staticmethod
    def _prepare_detection_summary(detection_summary, subset_indices):
        if isinstance(detection_summary, str):
            detection_summary = pd.read_csv(detection_summary, header=0)
        assert isinstance(detection_summary, pd.DataFrame)
        if subset_indices is None:
            subset_indices = list(range(len(detection_summary)))
        return detection_summary.iloc[subset_indices, :].reset_index(drop=True)

    def __init__(
        self, 
        detection_summary,
        data_root, 
        coco_annotations_path = 'annotations/instances_train2017.json', 
        images_path = 'train', 
        subset_indices=None, 
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [123.675/255, 116.28/255, 103.53/255], 
                [58.395/255, 57.12/255, 57.375/255])
        ])):
        '''
        - detector_summary: str or pandas dataframe: detections results
        - data_root: str: path to coco dataset
        - coco_annotations_path : str: path to coco_annotations, i.e.
        data_root/coco_annotations_path is overall path to the annotations
        - images_path : str : path to coco images, i.e.
        data_root/images_path is overall path to the images

        detection_summary is supposed to be have the following format:
        the first column is image_id 
        the rest columns are several properties of interest
        the detection_summary is supposed to have header
        '''
        super().__init__()
        self.detection_summary = self._prepare_detection_summary(detection_summary, subset_indices)
        self.data_root = data_root
        self.coco_annotations_path = coco_annotations_path
        self.images_path = images_path
        self.image_transform = image_transform

        with open(f"{self.data_root}/{self.coco_annotations_path}") as anno_file:
            annotations = json.load(anno_file)

        self.image_ids = self.detection_summary.iloc[:, 0].to_list()
        self.id2paths = {}
        self.id2properties = {}

        # set of allowed ids
        allowed_ids_set = set(self.image_ids)

        # create mapping image_id to file_names
        for image_data in annotations['images']:
            if image_data['id'] in allowed_ids_set:
                self.id2paths[image_data['id']] = image_data['file_name']

        # create mapping image_id to properties
        for i in range(len(self.detection_summary)):
            df_col = self.detection_summary.iloc[i]
            image_id = df_col.iloc[0]
            properties = df_col.iloc[1:].to_dict()
            self.id2properties[image_id] = properties

    def __len__(self):
        return len(self.image_ids)

    def properties_transform(self, idx):
        image_id = self.image_ids[idx]
        properties = self.id2properties[image_id]
        t_properties = torch.tensor(list(properties.values()))
        return t_properties

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(f"{self.data_root}/{self.images_path}/{self.id2paths[image_id]}")
        image = self.image_transform(image)
        properties = self.properties_transform(idx)
        return (image, properties)
