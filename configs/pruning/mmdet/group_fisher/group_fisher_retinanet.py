_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'


architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisher',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=architecture,
    batch_size=2,
    interval=10,
    delta='acts',
    save_ckpt_delta_thr = [0.75,0.5,0.25],
    mutator=dict(
        type='ChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer',tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio'))),
)
