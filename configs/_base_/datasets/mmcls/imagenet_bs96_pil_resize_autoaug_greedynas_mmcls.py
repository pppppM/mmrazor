_base_ = [
    'pipelines/rand_aug.py',
]
# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))


train_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment', 
        policies={{_base_.rand_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bilinear')),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        max_area_ratio=1 / 3,
        mode='rand',
        fill_color=tuple(img_norm_cfg['mean'][::-1]),
        fill_std=tuple(img_norm_cfg['std'][::-1])),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=96,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        ann_file='data/imagenet/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
