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
    
    @staticmethod
    def _im_name_to_id(name):
        return int(name.split('.')[0].lstrip('0'))

    @staticmethod
    def _parse_image_ids(detection_summary):
        names = detection_summary.iloc[:, 0].to_list()
        ids = [PolicyDataset._im_name_to_id(name) for name in names]
        return ids
    
    def check_detection_summary_consistency(self):
        assert len(self.wk_detection_summary) == len(self.sg_detection_summary)
        for i in range(len(self.wk_detection_summary)):
            assert self.wk_detection_summary.iloc[i, 0] == self.sg_detection_summary.iloc[i, 0]

    def __init__(
        self, 
        wk_detection_summary,
        sg_detection_summary,
        data_root, 
        statistic='mAP',
        coco_annotations_path = 'annotations/instances_train2017.json', 
        images_path = 'train', 
        subset_indices=None, 
        return_indices=False,
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [123.675/255, 116.28/255, 103.53/255], 
                [58.395/255, 57.12/255, 57.375/255])
        ])):
        '''
        - wk_detection_summary: str or pandas dataframe: detection results of weak model
        - sg_detection_summary: str or pandas dataframe: detection results of strong model
        - data_root: str: path to coco dataset
        - coco_annotations_path : str: path to coco_annotations, i.e.
        data_root/coco_annotations_path is overall path to the annotations
        - images_path : str : path to coco images, i.e.
        data_root/images_path is overall path to the images
        - statistic : str : statistic used for reward

        detection_summary is supposed to be have the following format:
        the first column is image_id 
        the rest columns are several properties of interest
        the detection_summary is supposed to have header
        '''
        super().__init__()
        self.wk_detection_summary =  self._prepare_detection_summary(
            wk_detection_summary, subset_indices)
        self.sg_detection_summary = self._prepare_detection_summary(
            sg_detection_summary, subset_indices)
        self.check_detection_summary_consistency()
        assert statistic in ['recall', 'mAP']
        self.statistic = statistic
        self.data_root = data_root
        self.coco_annotations_path = coco_annotations_path
        self.images_path = images_path
        self.image_transform = image_transform
        self.return_indices = return_indices

        with open(f"{self.data_root}/{self.coco_annotations_path}") as anno_file:
            annotations = json.load(anno_file)

        self.image_ids = self._parse_image_ids(self.wk_detection_summary)
        self.id2paths = {}
        self.id2wk_properties = {}
        self.id2sg_properties = {}

        # set of allowed ids
        allowed_ids_set = set(self.image_ids)

        # create mapping image_id to file_names
        for image_data in annotations['images']:
            if image_data['id'] in allowed_ids_set:
                self.id2paths[image_data['id']] = image_data['file_name']

        # create mapping image_id to properties
        for i in range(len(self.wk_detection_summary)):
            wk_df_col = self.wk_detection_summary.iloc[i]
            sg_df_col = self.sg_detection_summary.iloc[i]
            image_id = self._im_name_to_id(wk_df_col.iloc[0])
            self.id2wk_properties[image_id] = wk_df_col.iloc[1:].to_dict()
            self.id2sg_properties[image_id] = sg_df_col.iloc[1:].to_dict()

    def __len__(self):
        return len(self.image_ids)

    def properties_transform(self, idx):
        image_id = self.image_ids[idx]
        wk_metric = self.id2wk_properties[image_id][self.statistic]
        sg_metric = self.id2sg_properties[image_id][self.statistic]
        t_properties = torch.tensor([wk_metric, sg_metric])
        return t_properties
    
    def get_wk_statistics(self, idx):
        return self.id2wk_properties[self.image_ids[idx]]
    
    def get_sg_statistics(self, idx):
        return self.id2sg_properties[self.image_ids[idx]]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(f"{self.data_root}/{self.images_path}/{self.id2paths[image_id]}")
        image = self.image_transform(image)
        properties = self.properties_transform(idx)
        if not self.return_indices:
            return (image, properties)
        return (image, properties, torch.tensor(idx, dtype=torch.int32))
