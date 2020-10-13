from Auto_Augment.core.util import set_memory_growth
from Auto_Augment.core.util.augmentation_2d import apply_random_gamma
import tensorflow as tf
import numpy as np
import random


def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    val_length = int(len(x_train) * 0.2)
    x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
    x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train, test, val


class ConvClassifier(tf.keras.Model):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.model_layers = [
            tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    
    def call(self, x, training=False):
        for l in self.model_layers:
            if training:
                x = l(x)
            else:
                if type(l) is not tf.keras.layers.Dropout:
                    x = l(x)
        return x


def get_and_compile_model(model_func):
    model = model_func()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )
    return model


if __name__ == "__main__":
    batch_size = 32
    epochs = 2

    train, val, test = get_mnist()

    train = train.batch(batch_size)
    val = val.batch(batch_size)
    test = test.batch(batch_size)

    train = train.map(apply_random_gamma)

    model = get_and_compile_model(ConvClassifier)

    hist = model.fit(
        train,
        epochs=epochs,
        validation_data=val,
    )
    val_loss = hist.history['val_loss']
