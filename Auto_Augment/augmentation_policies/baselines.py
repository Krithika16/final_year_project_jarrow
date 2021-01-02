import tensorflow as tf


class NoAugmentationPolicy(tf.keras.Model):
    def __init__(self):
        super(NoAugmentationPolicy, self).__init__()

    def call(self, inputs, training=False):
        x, y = inputs
        return x, y


class RandomAugmentationPolicy(tf.keras.Model):
    def __init__(self):
        super(RandomAugmentationPolicy, self).__init__()

    def call(self, inputs, training=False):
        x, y = inputs
        return x, y


class FixAugmentationPolicy(tf.keras.Model):
    def __init__(self):
        super(FixAugmentationPolicy, self).__init__()

    def call(self, inputs, training=False):
        x, y = inputs
        return x, y
