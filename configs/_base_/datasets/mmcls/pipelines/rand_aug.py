resample = 'bilinear'
rand_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(-30, 30),
        interpolation='bicubic'),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(0, 4)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(0, 256)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0.1, 1.9)),
    dict(
        type='Contrast', magnitude_key='magnitude',
        magnitude_range=(0.1, 1.9)),
    dict(
        type='Brightness',
        magnitude_key='magnitude',
        magnitude_range=(0.1, 1.9)),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(0.1, 1.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        direction='horizontal',
        interpolation=resample),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        direction='vertical',
        interpolation=resample),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(-0.45, 0.45),
        direction='horizontal',
        interpolation=resample),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(-0.45, 0.45),
        direction='vertical',
        interpolation=resample)
]
