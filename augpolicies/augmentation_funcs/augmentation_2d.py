import random
import sys
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import array_ops

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
    assert tf.rank(image) == 4, f"NHWC format required, observed rank: {tf.rank(image)} for shape: {tf.shape(image)}"


# no aug


def apply_no_aug(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    *args,
    **kwargs,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    return image, label


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
    do_prob = do_prob * tf.constant(0.5)
    flips = tf.reshape(tf.random.categorical(tf.math.log([[1. - do_prob, do_prob]]), batch_size), (batch_size, 1, 1, 1))
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
    if mag > 0:
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
    if mag > 0:
        if tf.random.uniform(()) <= do_prob:
            lower = 0. if mag > 1. else 1. - mag
            image = tf.image.random_contrast(image, lower, 1. + mag + 0.001)
    return image, label


def add_space_domain_noise(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 1.0
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:
    pass


def apply_random_rotate(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 1.0,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    angles = tf.random.uniform((image.shape[0],), minval=-1., maxval=1.) * mag

    samples = tf.random.categorical(tf.math.log([[1 - do_prob, do_prob]]), image.shape[0])
    samples_mask = tf.cast(samples, tf.float32)

    angles = tf.math.multiply(angles, samples_mask).numpy()[0]

    image = tfa.image.rotate(image, angles)
    if apply_to_y:
        label = tfa.image.rotate(label, angles)
    return image, label


def apply_random_cutout(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float = 1.0,
    mag: float = 1.0,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    random_seed = random.randrange(sys.maxsize)
    enforce_rank(image)

    rank = len(image.shape)  # NHWC, HWC, HW
    if rank == 3:  # hwc
        h = image.shape[0]
        w = image.shape[1]
    elif rank == 4:  # nhwc
        h = image.shape[1]
        w = image.shape[2]

    cutout_w = int(h * 0.5 * mag)
    cutout_h = int(w * 0.5 * mag)
    cutout = min(cutout_w, cutout_h)
    if cutout % 2 == 1:
        cutout += 1

    if cutout_w > 0 and cutout_h > 0:
        tf.random.set_seed(random_seed)
        image = tfa.image.random_cutout(image, (cutout, cutout), constant_values=0, seed=random_seed)
        if apply_to_y:
            tf.random.set_seed(random_seed)
            label = tfa.image.random_cutout(label, (cutout, cutout), constant_values=0, seed=random_seed)
    return image, label


def apply_random_x_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    return apply_random_zoom(image, label, do_prob, mag, apply_to_y, x_mag=True, y_mag=False)


def apply_random_y_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    return apply_random_zoom(image, label, do_prob, mag, apply_to_y, x_mag=False, y_mag=True)


def apply_random_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
    x_mag: bool = True,
    y_mag: bool = True,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    mag = (1 - (tf.random.uniform((image.shape[0], 2)) - 0.5) * mag)
    ones_mask = tf.ones(mag.shape)

    samples_mask = tf.random.categorical(tf.math.log([[1 - do_prob, do_prob]]), mag.shape[0] * mag.shape[1])
    samples_mask = tf.reshape(samples_mask, mag.shape)
    mag = tf.where(samples_mask == 1, mag, ones_mask).numpy()

    if not x_mag:
        mag[:, 0] = 1.0
    if not y_mag:
        mag[:, 1] = 1.0

    return apply_zoom(image, label, mag, apply_to_y)


def apply_random_x_skew(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    return apply_random_skew(image, label, do_prob, mag, apply_to_y, x_mag=True, y_mag=False)


def apply_random_y_skew(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    return apply_random_skew(image, label, do_prob, mag, apply_to_y, x_mag=False, y_mag=True)


def apply_random_skew(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    do_prob: float,
    mag: float,
    apply_to_y: bool = False,
    x_mag: bool = True,
    y_mag: bool = True,
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    enforce_rank(image)
    mag = tf.random.uniform((image.shape[0], 2), minval=-1., maxval=1.) * mag

    samples = tf.random.categorical(tf.math.log([[1 - do_prob, do_prob]]), mag.shape[0] * mag.shape[1])
    samples_mask = tf.reshape(samples, mag.shape)
    samples_mask = tf.cast(samples_mask, tf.float32)

    mag = tf.math.multiply(mag, samples_mask).numpy()

    if not x_mag:
        mag[:, 0] = 0.
    if not y_mag:
        mag[:, 1] = 0.

    return apply_skew(image, label, mag, apply_to_y)


def apply_zoom(
    image: tfa.types.TensorLike,
    label: tfa.types.TensorLike,
    mag: tfa.types.TensorLike,
    apply_to_y: bool = False
) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

    if type(mag) is list or type(mag) is tuple:
        x_mag = mag[0]
        y_mag = mag[1]
    else:
        x_mag = mag[:, 0]
        y_mag = mag[:, 1]
    transforms = np.array([[1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0]], dtype=np.float32)
    rank = len(image.shape)  # NHWC, HWC, HW
    if rank == 3:  # hwc
        h = image.shape[0]
        w = image.shape[1]
    elif rank == 4:  # nhwc
        h = image.shape[1]
        w = image.shape[2]
        transforms = np.broadcast_to(transforms, [image.shape[0], 8]).copy()
    c = image.shape[-1]
    assert c == 1 or c == 3, "last column must be rgb or grayscale"

    x_mag = np.clip(x_mag, 0.05, None)
    y_mag = np.clip(y_mag, 0.05, None)

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

    if type(mag) is list or type(mag) is tuple:
        x_mag = mag[0]
        y_mag = mag[1]
    else:
        x_mag = mag[:, 0]
        y_mag = mag[:, 1]

    transforms = np.array([[1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0]], dtype=np.float32)
    rank = len(image.shape)  # NHWC, HWC, HW
    if rank == 3:  # hwc
        h = image.shape[0]
        w = image.shape[1]
    elif rank == 4:  # nhwc
        h = image.shape[1]
        w = image.shape[2]
        transforms = np.broadcast_to(transforms, [image.shape[0], 8]).copy()

    c = image.shape[-1]
    assert c == 1 or c == 3, "last column must be rgb or grayscale"

    x_mag = np.clip(x_mag, -2.5, 2.5)
    y_mag = np.clip(y_mag, -2.5, 2.5)

    transforms[:, 1] = x_mag
    transforms[:, 2] = -w * x_mag
    transforms[:, 3] = y_mag
    transforms[:, 5] = -h * y_mag

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


# prob, mag1, mag2 as an kwarg input

# def apply_random_quality(image: tfa.types.TensorLike, label: tfa.types.TensorLike, do_prob=0.1, min_quality=0, max_quality=100
# ) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:
#     if tf.random.uniform(()) <= do_prob:
#         v1 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         v2 = tf.random.uniform((), minval=min_quality, maxval=max_quality + 1, dtype=tf.int32)
#         image = tf.map_fn(fn=lambda t: tf.image.random_jpeg_quality(t, min(v1, v2), max(v1, v2)), elems=image)
#     return image, label

if __name__ == "__main__":
    (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., np.newaxis]
    x_train = x_train[:64]

    func = apply_random_skew

    import matplotlib.pyplot as plt

    mags = [0.0, 0.5, 1.0, 1.2]
    rs = 3
    f, axs = plt.subplots(rs * 2, len(mags) + 1)
    axs[0][0].imshow(x_train[0])

    for r in range(0, rs * 2):
        axs[r][0].axis('off')

    import time

    t0 = time.time()

    for idx, m in enumerate(mags):
        img, out_img = func(x_train, x_train, 1.0, m, apply_to_y=True)
        for r in range(rs):

            axs[r * 2][idx + 1].imshow(img[r])
            axs[r * 2][idx + 1].axis('off')
            # axs[r * 2 + 1][idx + 1].imshow(np.abs(img[r] - x_train[r]))
            axs[r * 2 + 1][idx + 1].imshow(out_img[r])
            axs[r * 2 + 1][idx + 1].axis('off')

    print(f"time took: {time.time() - t0:.2f}")
    plt.show()
