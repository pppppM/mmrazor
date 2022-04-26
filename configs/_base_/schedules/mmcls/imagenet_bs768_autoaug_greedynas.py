# optimizer
paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)
optimizer = dict(
    type='RMSpropTF',
    lr=0.048,
    eps=0.001,
    weight_decay=1e-5,
    momentum=0.9,
    paramwise_cfg=paramwise_cfg)optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='RazorStep',
    step=int(1669 * 2.4),
    gamma=0.97,
    per_epoch_iters=1669,
    by_epoch=False,
    warmup='linear',
    warmup_iters=3,
    warmup_ratio=1e-6 / 0.048,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=450)
