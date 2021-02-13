import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import sys
from typing import Callable, Tuple


# kwarg getter


def get_kwarg(
    id: str,
    mean: float,
    std: float = 0.0
) -> dict:
    return {
        id: np.random.normal(loc=mean, scale=std) if std else mean
    }


def kwargs_func_prob(
    do_prob_mean: float = 0.0,
    do_prob_std: float = 0.0
) -> Callable[[float, float], dict]:
    def prob_kwargs(
        do_prob_mean: float = do_prob_mean,
        do_prob_std: float = do_prob_std
    ) -> dict:
        return get_kwarg("do_prob", mean=do_prob_mean, std=do_prob_std)
    return prob_kwargs


def kwargs_func_prob_mag(
    do_prob_mean: float = 0.0,
    do_prob_std: float = 0.0,
    mag_mean: float = 0.0,
    mag_std: float = 0.0
) -> Callable[[float, float, float, float], dict]:
    def prob_mag_kwargs(
        do_prob_mean: float = do_prob_mean,
        do_prob_std: float = do_prob_std,
        mag_mean: float = mag_mean,
        mag_std: float = mag_std
    ) -> dict:
        return {
            **get_kwarg("do_prob", mean=do_prob_mean, std=do_prob_std),
            **get_kwarg("mag", mean=mag_mean, std=mag_std)
        }
    return prob_mag_kwargs


# only prob as an kwarg input

def apply_random_left_right_flip(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.2,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    image, label = flip_randomly_image_pair_2d(image, label, tf.image.random_flip_left_right, do_prob, apply_to_y)
    return image, label


def apply_random_up_down_flip(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.2,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    image, label = flip_randomly_image_pair_2d(image, label, tf.image.random_flip_up_down, do_prob, apply_to_y)
    return image, label


def flip_randomly_image_pair_2d(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    flip_op: Callable[[tfa.types.TensorLike, tfa.types.TensorLike, float, bool], Tuple[tfa.types.TensorLike, tfa.types.TensorLike]],
    do_prob: float,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    random_var = tf.random.uniform(()) <= do_prob
    image = tf.cond(pred=random_var,
                    true_fn=lambda: flip_op(image),
                    false_fn=lambda: image)
    if apply_to_y:
        label = tf.cond(pred=random_var,
                        true_fn=lambda: flip_op(label),
                        false_fn=lambda: label)
    return image, label


# prob, mag as an kwarg input

def apply_random_brightness(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.4,
    mag: float = 0.2
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if tf.random.uniform(()) <= do_prob:
        image = tf.image.random_brightness(image, mag)
    return image, label


def apply_random_hue(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.4,
    mag: float = 0.2
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if tf.random.uniform(()) <= do_prob:
        image = tf.image.random_hue(image, mag)
    return image, label


def apply_random_contrast(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 1.0
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if tf.random.uniform(()) <= do_prob:
        image = tf.image.random_contrast(image, 0.0, mag + 0.01)
    return image, label


def apply_random_shear(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 3.0,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if tf.random.uniform(()) <= do_prob:
        image = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_shear(t, mag), elems=image)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image, label


def apply_random_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = (2.0, 2.0),
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if type(mag) is float:
        mag = (mag, mag)
    if tf.random.uniform(()) <= do_prob:
        image = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_zoom(t, mag), elems=image)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image, label


# prob, mag1, mag2 as an kwarg input

# def apply_random_quality(image: tfa.types.TensorLike, label: tfa.types.TensorLike, do_prob=0.1, min_quality=0, max_quality=100) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:
#     if tf.random.uniform(()) <= do_prob:
#         v1 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         v2 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         image = tf.map_fn(fn=lambda t: tf.image.random_jpeg_quality(t, min(v1, v2), max(v1, v2)), elems=image)
#     return image, label
