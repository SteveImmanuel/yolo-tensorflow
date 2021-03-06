BASE_ARCHITECTURE = {
    'CONV_LAYERS':
        [
            (64, 7, 2),  # n_filters, filter_size, stride
            'M',  # maxpool 2x2 stride 2
            (192, 3, 1),
            'M',
            (128, 1, 1),
            (256, 3, 1),
            (256, 1, 1),
            (512, 3, 1),
            'M',
            [(256, 1, 1), (512, 3, 1), 4],
            (512, 1, 1),
            (1024, 3, 1),
            'M',
            [(512, 1, 1), (1024, 3, 1), 2],
            (1024, 3, 1),
            (1024, 3, 2),
            (1024, 3, 1),
            (1024, 3, 1),
        ],
    'FC_LAYER': 4096
}

FAST_ARCHITECTURE = {
    'CONV_LAYERS':
        [
            (32, 7, 2),
            'M',
            (96, 3, 1),
            'M',
            (64, 3, 1),
            (128, 3, 1),
            'M',
            (256, 3, 1),
            (512, 3, 1),
            'M',
            (512, 3, 1),
            (512, 3, 2),
            (512, 3, 1),
        ],
    'FC_LAYER': 256
}
