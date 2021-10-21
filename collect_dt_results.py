import os
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from mmdet.core.evaluation import eval_map, eval_recalls
from mmdet.datasets import CocoDataset
from mmdet.apis import init_detector, inference_detector


def compute_mAP(dt_results, gt_bboxes, gt_labels, iou_thr=0.5):
    image_annotations = [dict(bboxes=gt_bboxes, labels=gt_labels)]
    return eval_map([dt_results], image_annotations, iou_thr=iou_thr)[0]


def compute_recall(dt_results, gt_bboxes, iou_thr=0.5):
    proposals = np.concatenate(dt_results, axis=0)
    recalls = eval_recalls([gt_bboxes], [proposals], proposal_nums=len(gt_bboxes), iou_thrs=iou_thr)[0][0]
    if np.isnan(recalls):
        recalls = 0.0
    return recalls


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='data dir')
    parser.add_argument('-o', '--output_dir', type=str, help='output dir')
    parser.add_argument('-c', '--config_path', type=str, help='path to config')
    parser.add_argument('-w', '--checkpoint_path', type=str, help='path to checkpoint')
    parser.add_argument('--ignore_train', action='store_true')
    parser.add_argument('--ignore_val', action='store_true')
    return parser

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
        data_root=f'{args.data_dir}/train2017',
        ann_file=f'{args.data_dir}/annotations/instances_train2017.json',
        pipeline=processing_pipeline
    )

    val_dataset = CocoDataset(
        data_root=f'{args.data_dir}/val2017',
        ann_file=f'{args.data_dir}/annotations/instances_val2017.json',
        pipeline=processing_pipeline
    )

    # init detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = init_detector(args.config_path, checkpoint=args.checkpoint_path, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    image_file_names = []
    mAPs = []
    recalls = []

    if not args.ignore_train:
        # collect metrcis on the train_dataset
        for train_sample in tqdm(train_dataset):
            image = train_sample['img'].data.permute(1, 2, 0).numpy().astype(np.uint8)
            image_file_name = train_sample['img_metas'].data['ori_filename']
            gt_bboxes = train_sample['gt_bboxes'].data.numpy()
            gt_labels = train_sample['gt_labels'].data.numpy()

            dt_results = inference_detector(detector, image)[0]
            # compute mAP
            mAP = compute_mAP(dt_results, gt_bboxes, gt_labels)
            # compute Recalls
            recall = compute_recall(dt_results, gt_bboxes)

            image_file_names.append(image_file_name)
            mAPs.append(mAP)
            recalls.append(recall)

        dataframe = pd.DataFrame({
            "filename" : image_file_names,
            "mAP" : mAPs,
            "recall" : recalls
        })

        dataframe.to_csv(f"{args.output_dir}/train_metrics.csv")

    if not args.ignore_val:
        # collect metrcis on the val_dataset
        for val_sample in tqdm(val_dataset):
            image = val_sample['img'].data.permute(1, 2, 0).numpy().astype(np.uint8)
            image_file_name = val_sample['img_metas'].data['ori_filename']
            gt_bboxes = val_sample['gt_bboxes'].data.numpy()
            gt_labels = val_sample['gt_labels'].data.numpy()

            dt_results = inference_detector(detector, image)[0]
            # compute mAP
            mAP = compute_mAP(dt_results, gt_bboxes, gt_labels)
            # compute Recalls
            recall = compute_recall(dt_results, gt_bboxes)

            image_file_names.append(image_file_name)
            mAPs.append(mAP)
            recalls.append(recall)

        dataframe = pd.DataFrame({
            "filename" : image_file_names,
            "mAP" : mAPs,
            "recall" : recalls
        })

        dataframe.to_csv(f"{args.output_dir}/val_metrics.csv")

    print("Finished!")