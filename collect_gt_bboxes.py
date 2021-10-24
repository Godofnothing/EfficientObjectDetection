import os
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from mmdet.datasets import CocoDataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='data dir')
    parser.add_argument('-o', '--output_dir', type=str, help='output dir')
    parser.add_argument('--ignore_train', action='store_true')
    parser.add_argument('--ignore_val', action='store_true')
    return parser

def create_gt_bbox_data(dataset):
    res_list = {}
    # collect metrcis on the dataset
    for sample in tqdm(dataset):
        im_name = sample['img_metas'].data['ori_filename']
        im_id = int(im_name.split('.')[0].lstrip('0'))
        gt_bboxes = sample['gt_bboxes'].data.numpy()
        gt_labels = sample['gt_labels'].data.numpy()
        assert not im_id in res_list.keys()
        res_list[im_id] = (gt_bboxes, gt_labels)
    return res_list

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    processing_pipeline = [
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    train_dataset = CocoDataset(
        f'{args.data_dir}/annotations/instances_train2017.json', 
        processing_pipeline, 
        img_prefix=f'{args.data_dir}/train'
    )

    val_dataset = CocoDataset(
        f'{args.data_dir}/annotations/instances_val2017.json', 
        processing_pipeline, 
        img_prefix=f'{args.data_dir}/val'
    )

    os.makedirs(args.output_dir, exist_ok=True)

    image_file_names = []

    if not args.ignore_train:
        train_data = create_gt_bbox_data(train_dataset)
        with open(f"{args.output_dir}/train_gt.pkl", 'wb') as f:
            pickle.dump(res_list, f)

    if not args.ignore_val:
        res_list = create_gt_bbox_data(val_dataset)
        with open(f"{args.output_dir}/val_gt.pkl", 'wb') as f:
            pickle.dump(res_list, f)

    print("Finished!")