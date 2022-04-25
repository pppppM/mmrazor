# optimizer
optimizer = dict(
    type='RMSpropTF',
    lr=0.048,
    eps=0.001,
    weight_decay=1e-5,
    momentum=0.9,
    filter_bias_and_bn=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=int(1666 * 2.7),
    gamma=0.97,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1666 * 3,
    warmup_ratio=1e-6 / 0.048,
    warmup_by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=450 * 1666)
