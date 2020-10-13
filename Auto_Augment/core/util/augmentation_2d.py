import tensorflow as tf


def apply_random_gamma(image_tensor, label_tensor):
    do_gamma = tf.random.uniform([]) > 0.75
    gamma = tf.random.uniform([], 0.9, 1.1)
    gain = tf.random.uniform([], 0.95, 1.05)
    image_tensor = tf.cond(do_gamma, lambda: tf.image.adjust_gamma(image_tensor, gamma=gamma, gain=gain), lambda: image_tensor)
    return image_tensor, label_tensor
