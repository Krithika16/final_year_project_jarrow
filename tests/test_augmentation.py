import pytest
import tensorflow as tf


@pytest.mark.parametrize(
    "e_total, aug_apps, expected_output",
    [
        (5, 0, []),
        (5, 1, [4]),
        (5, 2, [2, 4]),
        (5, 3, [1, 2, 4]),
        (5, 4, [0, 2, 3, 4]),
        (5, 5, [0, 1, 2, 3, 4]),
        (10, 10, [0,1,2,3,4,5,6,7,8,9]),
        (10,  9, [  1,2,3,4,5,6,7,8,9]),
        (10,  8, [  1,  3,4,5,6,7,8,9]),
        (10,  7, [  1,  3,  5,6,7,8,9]),
        (10,  6, [  1,  3,  5,  7,8,9]),
        (10,  5, [  1,  3,  5,  7,  9]),
        (10,  4, [    2,  4,    7,  9]),
        (10,  3, [    2,      6,    9]),
        (10,  2, [        4,        9]),
        (10,  1, [                  9]),
        (10,  0, [])
    ]
)
def test_half_aug_policy_idxes(e_total, aug_apps, expected_output):
    def temp():
        return None

    from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy

    pol = HalfAugmentationPolicy([temp, temp], None, e_total, aug_apps, interval=True)

    idxes = pol.get_interval_idxes(e_total, aug_apps)
    print(idxes, "==", expected_output)
    assert idxes == expected_output


def test_augs_import():
    import augpolicies.augmentation_funcs.augmentation_1d
    import augpolicies.augmentation_funcs.augmentation_2d


@pytest.mark.parametrize("func", ['apply_random_left_right_flip', 'apply_random_up_down_flip'])
@pytest.mark.parametrize('batch', [True])
@pytest.mark.parametrize('prob', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('apply_to_y', [True, False])
@pytest.mark.parametrize('c', [1, 3])
@pytest.mark.parametrize('dataset', [tf.keras.datasets.fashion_mnist, tf.keras.datasets.cifar10])
def test_random_aug_func_with_apply_to_y_no_mag(func, batch, prob, apply_to_y, c, dataset):
    import importlib
    import tensorflow as tf
    import numpy as np
    m = importlib.import_module('augpolicies.augmentation_funcs.augmentation_2d')
    func = getattr(m, func)
    (x_train, _), (_, _) = dataset.load_data()
    r = 10 if apply_to_y else 1
    for _ in range(r):
        if batch:
            data = x_train[:64]
        else:
            data = x_train[0]
        if c:
            if len(data.shape) == 3:
                data = data[..., np.newaxis]
                if c == 3:
                    target_shape = list(data.shape)
                    target_shape[-1] = c
                    data = np.broadcast_to(data, target_shape)
            elif len(data.shape) == 4:
                s = data.shape
                if c == s[-1]:
                    pass
                elif c == 3 & s[-1] == 1:
                    target_shape = list(data.shape)
                    target_shape[-1] = c
                    data = np.broadcast_to(data, target_shape)
                elif c == 1 & s[-1] == 3:
                    data = tf.math.reduce_mean(data, axis=-1, keepdims=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        trans, lab = func(data, data, do_prob=tf.constant(prob), apply_to_y=apply_to_y)
        assert data.shape == trans.shape
        if apply_to_y:
            assert trans.shape == lab.shape
            assert np.array_equal(trans, lab)


@pytest.mark.parametrize("func", ['apply_random_zoom', 'apply_random_skew', 'apply_random_rotate', 'apply_random_cutout'])
@pytest.mark.parametrize('batch', [True])
@pytest.mark.parametrize('prob', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('mag', [0, 0.5, 1.0])
@pytest.mark.parametrize('apply_to_y', [True, False])
@pytest.mark.parametrize('c', [1, 3])
@pytest.mark.parametrize('dataset', [tf.keras.datasets.fashion_mnist, tf.keras.datasets.cifar10])
def test_random_aug_func_with_apply_to_y(func, batch, prob, mag, apply_to_y, c, dataset):
    import importlib
    import tensorflow as tf
    import numpy as np
    m = importlib.import_module('augpolicies.augmentation_funcs.augmentation_2d')
    func = getattr(m, func)
    (x_train, _), (_, _) = dataset.load_data()
    r = 10 if apply_to_y else 1
    for _ in range(r):
        if batch:
            data = x_train[:64]
        else:
            data = x_train[0]
        if c:
            if len(data.shape) == 3:
                data = data[..., np.newaxis]
                if c == 3:
                    target_shape = list(data.shape)
                    target_shape[-1] = c
                    data = np.broadcast_to(data, target_shape)
            elif len(data.shape) == 4:
                s = data.shape
                if c == s[-1]:
                    pass
                elif c == 3 & s[-1] == 1:
                    target_shape = list(data.shape)
                    target_shape[-1] = c
                    data = np.broadcast_to(data, target_shape)
                elif c == 1 & s[-1] == 3:
                    data = tf.math.reduce_mean(data, axis=-1, keepdims=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        trans, lab = func(data, data, do_prob=prob, mag=mag, apply_to_y=apply_to_y)
        assert data.shape == trans.shape
        if apply_to_y:
            assert trans.shape == lab.shape
            assert np.array_equal(trans, lab)


@pytest.mark.parametrize("func", ['apply_random_brightness', 'apply_random_contrast'])
@pytest.mark.parametrize('batch', [True])
@pytest.mark.parametrize('prob', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('mag', [0, 0.5, 1.0])
@pytest.mark.parametrize('c', [1, 3])
@pytest.mark.parametrize('dataset', [tf.keras.datasets.fashion_mnist, tf.keras.datasets.cifar10])
def test_random_aug_func(func, batch, prob, mag, c, dataset):
    import importlib
    import tensorflow as tf
    import numpy as np
    m = importlib.import_module('augpolicies.augmentation_funcs.augmentation_2d')
    func = getattr(m, func)

    (x_train, _), (_, _) = dataset.load_data()
    if batch:
        data = x_train[:32]
    else:
        data = x_train[0]
    if c:
        if len(data.shape) == 3:
            data = data[..., np.newaxis]
            if c == 3:
                target_shape = list(data.shape)
                target_shape[-1] = c
                data = np.broadcast_to(data, target_shape)
        elif len(data.shape) == 4:
            s = data.shape
            if c == s[-1]:
                pass
            elif c == 3 & s[-1] == 1:
                target_shape = list(data.shape)
                target_shape[-1] = c
                data = np.broadcast_to(data, target_shape)
            elif c == 1 & s[-1] == 3:
                data = tf.math.reduce_mean(data, axis=-1, keepdims=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    trans, _ = func(data, data, do_prob=prob, mag=mag)
    assert data.shape == trans.shape
