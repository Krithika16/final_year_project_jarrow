import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import sys


# kwarg getter


def get_kwarg(id, mean, std=0.0):
    return {
        id: np.random.normal(loc=mean, scale=std) if std else mean
    }


def kwargs_func_prob(do_prob_mean=0.0, do_prob_std=0.0):
    def prob_kwargs(do_prob_mean=do_prob_mean, do_prob_std=do_prob_std):
        return get_kwarg("do_prob", mean=do_prob_mean, std=do_prob_std)
    return prob_kwargs


def kwargs_func_prob_mag(do_prob_mean=0.0, do_prob_std=0.0,
                         mag_mean=0.0, mag_std=0.0):
    def prob_mag_kwargs(do_prob_mean=do_prob_mean, do_prob_std=do_prob_std,
                        mag_mean=mag_mean, mag_std=mag_std):
        return {
            **get_kwarg("do_prob", mean=do_prob_mean, std=do_prob_std),
            **get_kwarg("mag", mean=mag_mean, std=mag_std)
        }
    return prob_mag_kwargs


# only prob as an kwarg input

def apply_random_left_right_flip(image_tensor, label_tensor, do_prob=0.2, apply_to_y=False):
    image_tensor, label_tensor = flip_randomly_image_pair_2d(image_tensor, label_tensor, tf.image.random_flip_left_right, do_prob, apply_to_y)
    return image_tensor, label_tensor


def apply_random_up_down_flip(image_tensor, label_tensor, do_prob=0.2, apply_to_y=False):
    image_tensor, label_tensor = flip_randomly_image_pair_2d(image_tensor, label_tensor, tf.image.random_flip_up_down, do_prob, apply_to_y)
    return image_tensor, label_tensor


def flip_randomly_image_pair_2d(image_tensor, label_tensor, flip_op, do_prob, apply_to_y):
    random_var = tf.random.uniform(()) <= do_prob
    image_tensor = tf.cond(pred=random_var,
                           true_fn=lambda: flip_op(image_tensor),
                           false_fn=lambda: image_tensor)
    if apply_to_y:
        label_tensor = tf.cond(pred=random_var,
                               true_fn=lambda: flip_op(label_tensor),
                               false_fn=lambda: label_tensor)
    return image_tensor, label_tensor


# prob, mag as an kwarg input

def apply_random_brightness(image_tensor, label_tensor, do_prob=0.4, mag=0.2):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_brightness(image_tensor, mag)
    return image_tensor, label_tensor


def apply_random_hue(image_tensor, label_tensor, do_prob=0.4, mag=0.2):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_hue(image_tensor, mag)
    return image_tensor, label_tensor


def apply_random_contrast(image_tensor, label_tensor, do_prob=1.0, mag=1.0):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_contrast(image_tensor, 0.0, mag + 0.01)
    return image_tensor, label_tensor


def apply_random_shear(image_tensor, label_tensor, do_prob=1.0, mag=3.0, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_shear(t, mag), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_zoom(image_tensor, label_tensor, do_prob=1.0, mag=(2.0, 2.0), apply_to_y=False):
    if type(mag) is float:
        mag = (mag, mag)
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_zoom(t, mag), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


# prob, mag1, mag2 as an kwarg input

# def apply_random_quality(image_tensor, label_tensor, do_prob=0.1, min_quality=0, max_quality=100):
#     if tf.random.uniform(()) <= do_prob:
#         v1 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         v2 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         image_tensor = tf.map_fn(fn=lambda t: tf.image.random_jpeg_quality(t, min(v1, v2), max(v1, v2)), elems=image_tensor)
#     return image_tensor, label_tensor
