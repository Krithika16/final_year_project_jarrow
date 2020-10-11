from Auto_Augment.core import set_memory_growth
import tensorflow as tf
import numpy as np


def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    val_length = int(len(x_train) * 0.2)
    x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
    x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_and_compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


# def train_model(train_data, val_data)

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_mnist()

    model = get_and_compile_model()

    # add custom training loop

    hist = model.fit(
        x_train,
        y_train,
        epochs=2,
        validation_data=(x_val, y_val),
    )

    val_loss = hist.history['val_loss']
