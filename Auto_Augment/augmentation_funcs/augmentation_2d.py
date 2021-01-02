import tensorflow as tf


def apply_random_brightness(image_tensor, label_tensor, do_prob=1.0, magnitude=1.0, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_brightness(image_tensor, magnitude)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_left_right_flip(image_tensor, label_tensor, do_prob=1.0, magnitude=None, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_flip_left_right(image_tensor, magnitude)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_up_down_flip(image_tensor, label_tensor, do_prob=1.0, magnitude=None, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_flip_up_down(image_tensor, magnitude)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_shear(image_tensor, label_tensor, do_prob=1.0, magnitude=3.0, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_shear(t, magnitude), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_zoom(image_tensor, label_tensor, do_prob=1.0, magnitude=(2.0, 2.0), apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.map_fn(fn=lambda t: tf.keras.preprocessing.image.random_zoom(t, magnitude), elems=image_tensor)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor


def apply_random_contrast(image_tensor, label_tensor, do_prob=1.0, magnitude=None, apply_to_y=False):
    if tf.random.uniform(()) <= do_prob:
        image_tensor = tf.image.random_contrast(image_tensor, 0.0, 1.0)
        if apply_to_y:
            raise NotImplementedError("Need to get the arguments to feed into both x and y")
    return image_tensor, label_tensor
