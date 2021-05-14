import numpy as np


def make_3d(dataset):
    x, y = dataset
    assert len(x.shape) == 4
    if x.shape[-1] != 3:
        assert x.shape[-1] == 1, f"The channel shape should be grayscale (channel = 1), current value: {x.shape[-1]}"
        x = np.broadcast_to(x, (*x.shape[:-1], 3))
    return (x, y)


def pad_to_min_size(dataset, min_size):
    x, y = dataset
    print(type(x))
    assert len(x.shape) == 4
    s = x.shape
    h, w = s[1], s[2]

    changed = False
    top_pad, bottom_pad = 0, 0
    left_pad, right_pad = 0, 0

    if h < min_size[0]:
        # make taller
        changed = True
        required_taller = min_size[0] - h
        top_pad = required_taller // 2
        bottom_pad = required_taller - top_pad

    if w < min_size[1]:
        # make wider
        changed = True
        required_wider = min_size[1] - w
        left_pad = required_wider // 2
        right_pad = required_wider - left_pad

    if changed:
        paddings = [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]]
        x = np.pad(x, paddings)
    return (x, y)
