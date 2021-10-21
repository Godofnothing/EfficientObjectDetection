NUM_CLASSES = 80
MMDET_DIR = '/trinity/home/d.kuznedelev/.conda/envs/mmlab/lib/python3.7/site-packages/mmdet'
DATA_DIR = '/trinity/home/d.kuznedelev/Datasets/COCO'

NUM_GPU = 1
NUM_WORKERS = 1
SAMPLES_PER_GPU = 128

_base_ = f'{MMDET_DIR}/configs/yolox/yolox_tiny_8x8_300e_coco.py'

# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (160, 160)


train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MinIoURandomCrop'),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=f'{DATA_DIR}/annotations/instances_train2017.json',
        img_prefix=f'{DATA_DIR}/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale
)

val_dataset = dict(
    type=dataset_type,
    ann_file=f'{DATA_DIR}/annotations/instances_val2017.json',
    img_prefix=f'{DATA_DIR}/val2017/',
    pipeline=test_pipeline
)

data = dict(
    _delete_=True,
    samples_per_gpu=SAMPLES_PER_GPU,
    workers_per_gpu=NUM_WORKERS,
    train=train_dataset,
    val=val_dataset
)

resume_from = None
interval = 1

# Execute in the order of insertion when the priority is the same.
# The smaller the value, the higher the priority
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(10, 20),
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]

optimizer = dict(
    _delete_=True,
    type='Adam', 
    lr=0.003 * NUM_GPU * SAMPLES_PER_GPU / 64, 
    weight_decay=0.00001
)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=interval)
evaluation = dict(interval=interval, metric='bbox')

load_from = "work_dirs/yolox_tiny_160_coco/latest.pth"
