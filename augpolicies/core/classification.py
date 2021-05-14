from augpolicies.augmentation_policies.baselines import \
    NoAugmentationPolicy, AugmentationPolicy
import tensorflow as tf
import numpy as np
import random


def get_classification_data(dataset=tf.keras.datasets.fashion_mnist.load_data, val_split=0.1, normalise_factor=255.0):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    if len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
    x_train = x_train / normalise_factor
    if len(x_test.shape) == 3:
        x_test = x_test[..., np.newaxis]
    x_test = x_test / normalise_factor
    if val_split is not None:
        val_length = int(len(x_train) * val_split)
        x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
        x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]
    else:
        x_val = y_val = None

    x_train = np.float32(x_train)
    x_val = np.float32(x_val)
    x_test = np.float32(x_test)

    train = (x_train, y_train)
    val = (x_val, y_val)
    test = (x_test, y_test)
    return train, val, test


def get_and_compile_model(model_func, lr=0.001, from_logits=True):
    model = model_func()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=['accuracy'],
    )
    return model


class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.requires_3d = False
        self.min_size = (28, 28)
        self.m_ = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs, training=False):
        return self.m_.call(inputs, training=training)


class EfficientNetB0(tf.keras.Model):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.requires_3d = True
        self.min_size = (32, 32)
        self.m_ = tf.keras.models.Sequential([
            tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=None,
                                                 input_shape=None, pooling='max', classes=1000,
                                                 classifier_activation='softmax'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])

    def call(self, inputs, training=False):
        return self.m_.call(inputs, training=training)


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.requires_3d = False
        self.min_size = (28, 28)
        self.m_ = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    def call(self, inputs, training=False):
        return self.m_.call(inputs, training=training)


class SimpleModel_Softmax(tf.keras.Model):
    def __init__(self):
        super(SimpleModel_Softmax, self).__init__()
        self.requires_3d = False
        self.min_size = (28, 28)
        self.m_ = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs, training=False):
        return self.m_.call(inputs, training=training)


