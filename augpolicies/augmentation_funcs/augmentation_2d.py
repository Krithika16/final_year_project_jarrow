import tensorflow as tf
from tensorflow.python.ops import array_ops
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


def enforce_rank(image):
    assert tf.rank(image).numpy() == 4, "NHWC format required"


# only prob as an kwarg input


def apply_random_left_right_flip(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.2,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    image, label = flip_randomly_image_pair(image, label, 1, do_prob, apply_to_y)
    return image, label


def apply_random_up_down_flip(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.2,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    image, label = flip_randomly_image_pair(image, label, 0, do_prob, apply_to_y)
    return image, label


def flip_randomly_image_pair(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    flip_index: int,
    do_prob: float,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    batch_size = image.shape[0]
    flips = tf.reshape(tf.random.categorical(tf.math.log([[1 - do_prob, do_prob]]), batch_size), (batch_size, 1, 1, 1))
    flips = tf.cast(flips, image.dtype)
    flipped_input = array_ops.reverse(image, [flip_index + 1])
    image = flips * flipped_input + (1 - flips) * image
    if apply_to_y:
        flipped_input = array_ops.reverse(label, [flip_index + 1])
        label = flips * flipped_input + (1 - flips) * label
    return image, label


# prob, mag as an kwarg input

def apply_random_brightness(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.4,
    mag: float = 0.2
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    if tf.random.uniform([]) <= do_prob:
        image = tf.image.random_brightness(image, mag)
    return image, label


def apply_random_hue(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 0.4,
    mag: float = 0.2
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    if tf.random.uniform([]) <= do_prob:
        image = tf.image.random_hue(image, mag)
    return image, label


def apply_random_contrast(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 1.0
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    if tf.random.uniform(()) <= do_prob:
        image = tf.image.random_contrast(image, 1 - mag, 1 + mag)
    return image, label


def apply_random_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    mag = (1 - (tf.random.uniform((image.shape[0], 2)) - 0.5) * mag)
    return apply_zoom(image, label, mag, apply_to_y)


def apply_random_skew(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    mag = tf.random.uniform((image.shape[0], 2), minval=-1., maxval=1.) * mag
    return apply_skew(image, label, mag, apply_to_y)


# def apply_random_mean_filter(
#     image: tfa.types.TensorLike,
#     label: tfa.types.TensorLike,
#     do_prob: float,
#     mag: float,
#     apply_to_y: bool = False
# ) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

#     enforce_rank(image)
#     batch_size = image.shape[0]
#     rv = tf.random.categorical(tf.math.log([[1 - do_prob, do_prob]]), batch_size)
#     rv = tf.reshape(rv, (batch_size, 1, 1, 1))
#     # rv = tf.broadcast_to(rv, image.shape)
#     image = tf.where(rv, tfa.image.mean_filter2d(image, filter_shape=int(mag)), image)
#     if apply_to_y:
#         label = tf.cast(label, tf.float32)
#         label = tf.where(rv, tfa.image.mean_filter2d(label, filter_shape=(mag, mag)), label)
#     return image, label


def apply_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    mag: tfa.types.TensorLike,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if type(mag) is tuple or type(mag) is list:
        x_mag = mag[0]
        y_mag = mag[1]
    else:
        x_mag = mag[:, 0]
        y_mag = mag[:, 1]
    transforms = np.array([[1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0]])
    rank = tf.rank(image).numpy()  # NHWC, HWC, HW
    if rank == 3:  # hwc
        h = image.shape[0]
        w = image.shape[1]
    elif rank == 4:  # nhwc
        h = image.shape[1]
        w = image.shape[2]
        transforms = np.broadcast_to(transforms, [image.shape[0], 8])
    c = image.shape[-1]
    assert c == 1 or c == 3, "last column must be rgb or grayscale"

    transforms.setflags(write=1)
    transforms[:, 0] = 1.0 / x_mag
    transforms[:, 2] = -(w * (1 - x_mag)) / (2 * x_mag)
    transforms[:, 4] = 1.0 / y_mag
    transforms[:, 5] = -(h * (1 - y_mag)) / (2 * y_mag)
    if rank == 3:
        transforms = transforms[0]
    image = tfa.image.transform(image, transforms)
    if apply_to_y:
        label = tfa.image.transform(label, transforms)
    return image, label


def apply_skew(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    mag: tfa.types.TensorLike,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if type(mag) is tuple or type(mag) is list:
        x_mag = mag[0]
        y_mag = mag[1]
    else:
        x_mag = mag[:, 0]
        y_mag = mag[:, 1]
    transforms = np.array([[1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0]])
    rank = tf.rank(image)  # NHWC, HWC, HW
    if rank == 4:  # nhwc
        transforms = np.broadcast_to(transforms, [image.shape[0], 8])
    c = image.shape[-1]
    assert c == 1 or c == 3, "last column must be rgb or grayscale"

    x_mag = tf.clip_by_value(x_mag, -2.5, 2.5)
    y_mag = tf.clip_by_value(y_mag, -2.5, 2.5)
    transforms.setflags(write=1)
    transforms[:, 1] = x_mag
    transforms[:, 2] = -250 * x_mag
    transforms[:, 3] = y_mag
    transforms[:, 5] = -250 * y_mag
    if rank == 3:
        transforms = transforms[0]

    image = tfa.image.transform(image, transforms)
    if apply_to_y:
        label = tfa.image.transform(label, transforms)
    return image, label


# def apply_weird_y():
#     factor = -3e-3
#     flipped_bw = tf.image.random_flip_left_right(bw_img_rgb)
#     flipped_bw = tf.image.random_flip_up_down(flipped_bw)
#     transform = tfa.image.transform(flipped_bw, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, factor * 1.0, factor * 1.0])
#     transform = tf.image.random_flip_left_right(transform)
#     transform = tf.image.random_flip_up_down(transform)
#     _ = plt.imshow(transform[0])
