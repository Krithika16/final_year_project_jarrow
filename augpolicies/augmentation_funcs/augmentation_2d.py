import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import sys


def apply_random_brightness(image_tensor, label_tensor, do_prob=0.4, max_delta=0.2):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_brightness(image_tensor, max_delta)
    return image_tensor, label_tensor


def apply_random_hue(image_tensor, label_tensor, do_prob=0.4, max_delta=0.2):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_hue(image_tensor, max_delta)
    return image_tensor, label_tensor


def apply_random_quality(image_tensor, label_tensor, do_prob=0.1, min_quality=0, max_quality=100):
    if tf.random.uniform(()) <= do_prob:
        v1 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
        v2 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
        image_tensor = tf.map_fn(fn=lambda t: tf.image.random_jpeg_quality(t, min(v1, v2), max(v1, v2)), elems=image_tensor)
    return image_tensor, label_tensor


def apply_random_contrast(image_tensor, label_tensor, do_prob=1.0, magnitude=1.0):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_contrast(image_tensor, 0.0, magnitude + 0.01)
    return image_tensor, label_tensor


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


def apply_random_shear(image_tensor, label_tensor, do_prob=1.0, magnitude=3.0, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_shear(t, magnitude), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_zoom(image_tensor, label_tensor, do_prob=1.0, magnitude=(2.0, 2.0), apply_to_y=False):
    if type(magnitude) is float:
        magnitude = (magnitude, magnitude)
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_zoom(t, magnitude), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor
