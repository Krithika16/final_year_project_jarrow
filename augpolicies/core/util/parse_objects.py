import tensorflow as tf
from augpolicies.core.classification import (SimpleModel, ConvModel, EfficientNetB0)
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom, \
    apply_random_x_skew, apply_random_y_skew, apply_random_x_zoom, apply_random_y_zoom, apply_random_rotate, apply_random_cutout


def parse_dataset(dataset_str):
    if dataset_str == 'cifar10':
        dataset = tf.keras.datasets.cifar10
    elif dataset_str == 'fmnist':
        dataset = tf.keras.datasets.fashion_mnist
    return dataset

def parse_model(model_str):
    if model_str == 'SimpleModel':
        model = SimpleModel
    elif model_str == 'ConvModel':
        model = ConvModel
    elif model_str == 'EfficientNetB0':
        model = EfficientNetB0
    return model


def parse_aug(aug_str):
    if aug_str == "apply_random_left_right_flip":
        aug = apply_random_left_right_flip
    elif aug_str == "apply_random_up_down_flip":
        aug = apply_random_up_down_flip
    elif aug_str == "apply_random_contrast":
        aug = apply_random_contrast
    elif aug_str == "apply_random_skew":
        aug = apply_random_skew
    elif aug_str == "apply_random_zoom":
        aug = apply_random_zoom
    elif aug_str == "apply_random_x_skew":
        aug = apply_random_x_skew
    elif aug_str == "apply_random_y_skew":
        aug = apply_random_y_skew
    elif aug_str == "apply_random_x_zoom":
        aug = apply_random_x_zoom
    elif aug_str == "apply_random_y_zoom":
        aug = apply_random_y_zoom
    elif aug_str == "apply_random_brightness":
        aug = apply_random_brightness
    elif aug_str == "apply_random_rotate":
        aug = apply_random_rotate
    elif aug_str == "apply_random_cutout":
        aug = apply_random_cutout
    return aug

def parse_list(strs, parse_func):
    objects = []
    for object_str in strs:
        objects.append(parse_func(object_str))
    return objects
