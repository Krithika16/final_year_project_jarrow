import tensorflow as tf
from tensorflow.keras.datasets import cifar10, fashion_mnist
from augpolicies.core.util.system_hardware import get_strategy_for_system
from augpolicies.core.classification import (SimpleModel, ConvModel, EfficientNetB0)
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom, \
    apply_random_x_skew, apply_random_y_skew, apply_random_x_zoom, apply_random_y_zoom, apply_random_rotate, apply_random_cutout


def parse_dataset(dataset_str):
    if dataset_str == 'cifar10':
        return cifar10
    elif dataset_str == 'fmnist':
        return fashion_mnist
    else:
        raise ValueError(f"dataset_str {dataset_str} is not recognised,")


def parse_model(model_str):
    if model_str == 'SimpleModel':
        return SimpleModel
    elif model_str == 'ConvModel':
        return ConvModel
    elif model_str == 'EfficientNetB0':
        return EfficientNetB0
    else:
        raise ValueError(f"model_str {model_str} is not recognised,")


def parse_aug(aug_str):
    if aug_str == "apply_random_left_right_flip":
        return apply_random_left_right_flip
    elif aug_str == "apply_random_up_down_flip":
        return apply_random_up_down_flip
    elif aug_str == "apply_random_contrast":
        return apply_random_contrast
    elif aug_str == "apply_random_skew":
        return apply_random_skew
    elif aug_str == "apply_random_zoom":
        return apply_random_zoom
    elif aug_str == "apply_random_x_skew":
        return apply_random_x_skew
    elif aug_str == "apply_random_y_skew":
        return apply_random_y_skew
    elif aug_str == "apply_random_x_zoom":
        return apply_random_x_zoom
    elif aug_str == "apply_random_y_zoom":
        return apply_random_y_zoom
    elif aug_str == "apply_random_brightness":
        return apply_random_brightness
    elif aug_str == "apply_random_rotate":
        return apply_random_rotate
    elif aug_str == "apply_random_cutout":
        return apply_random_cutout
    else:
        raise ValueError(f"aug_str {aug_str} is not recognised,")


def parse_list(strs, parse_func):
    objects = []
    for object_str in strs:
        objects.append(parse_func(object_str))
    return objects


def parse_strategy(strat_str):
    if strat_str is None:
        return get_strategy_for_system()
    elif strat_str == "default":
        return tf.distribute.get_strategy()
    elif strat_str == "gpu1":
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    elif strat_str == "gpu2":
        return tf.distribute.OneDeviceStrategy("/gpu:1")
    elif strat_str == "multi-gpu":
        return tf.distribute.MirroredStrategy()
    else:
        raise ValueError(f"strat_str {strat_str} is not recognised,")
