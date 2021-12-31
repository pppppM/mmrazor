_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs256_autoslim.py',
    '../../_base_/schedules/mmcls/imagenet_bs2048_autoslim.py',
    '../../_base_/mmcls_runtime.py'
]


# model settings
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


algorithm = dict(
    type='BaseAlgorithm',
    architecture=dict(type='MMClsArchitecture', model=model),
    pruner=dict(
        type='ResRepPruner',
        flops_constraint=3000000),
    retraining=False)
