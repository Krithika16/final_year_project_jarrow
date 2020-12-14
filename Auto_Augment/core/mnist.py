from Auto_Augment.core.util import set_memory_growth
from Auto_Augment.core.util.augmentation_2d import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    val_length = int(len(x_train) * 0.2)
    x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
    x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]

    x_train = np.float32(x_train)
    x_val = np.float32(x_val)
    x_test = np.float32(x_test)

    train = (x_train, y_train)
    val = (x_val, y_val)
    test = (x_test, y_test)
    return train, test, val


def data_generator(x, y, batch_size=32, train=True):
    x_len = len(x)
    idxes = np.array(list(range(x_len)))
    np.random.shuffle(idxes)
    idx = 0
    while idx + batch_size < (x_len - 1):
        x_ = x[idx: idx+batch_size]
        y_ = y[idx: idx+batch_size]
        yield (x_, y_)
        idx += batch_size


class ConvClassifier(tf.keras.Model):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.model_layers = [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1), batch_size=128),
            # tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
            # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            # tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
            # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

    def call(self, x, training=False):
        for lay in self.model_layers:
            x = lay(x)
        return x


def get_and_compile_model(model_func):
    model = model_func()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'],
    )
    return model


@tf.function
def train_step(model, inputs, targets, optimizer, loss_func):
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss = loss_func(targets, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, pred


@tf.function
def val_step(model, inputs, targets, loss_func):
    pred = model(inputs, training=False)
    loss = loss_func(targets, pred)
    return loss, pred


def supervised_train_loop(model, train, val, batch_size=128, epochs=20):
    train_loss_results = []
    train_val_loss_results = []
    train_acc_results = []
    train_val_acc_results = []

    train_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*train, batch_size, True))
    val_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*val, batch_size, False))

    optimizer = model.optimizer
    loss = model.loss

    for e in tqdm(range(epochs)):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        tf.print(f"{e+1:03d}/{epochs:03d}: ", end="")

        for x, y in train_ds:
            x, y = augmentation_policy(x, y)
            tr_loss, tr_pred = train_step(model, x, y, optimizer, loss)
            epoch_loss_avg.update_state(tr_loss)
            epoch_acc.update_state(y, tr_pred)

        for x, y in val_ds:
            val_loss, val_pred = val_step(model, x, y, loss)
            epoch_val_loss_avg.update_state(val_loss)
            epoch_val_acc.update_state(y, val_pred)

        train_loss_results.append(epoch_loss_avg.result())
        train_val_loss_results.append(epoch_val_loss_avg.result())
        train_acc_results.append(epoch_acc.result())
        train_val_acc_results.append(epoch_val_acc.result())

        tf.print(f"Loss: {train_loss_results[-1]:.3f}, Val Loss: {train_val_loss_results[-1]:.3f}, Acc: {train_acc_results[-1]:.3f}, Val Acc: {train_val_acc_results[-1]:.3f}")
    return train_loss_results, train_val_loss_results, train_acc_results, train_val_acc_results


def augmentation_policy(x, y):
    # x, y = apply_random_left_right_flip(x, y)
    # x, y = apply_random_up_down_flip(x, y)
    # x, y = apply_random_shear(x, y)
    # x, y = apply_random_zoom(x, y)
    # x, y = apply_random_brightness(x, y)
    # x, y = apply_random_contrast(x, y)
    return x, y


if __name__ == "__main__":

    train, val, test = get_mnist()
    train_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*train, 128, True))
    val_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*val, 128, False))

    model = get_and_compile_model(ConvClassifier)
    supervised_train_loop(model, train, val)

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #     tf.keras.layers.Dense(128,activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # model.compile(
    #     loss='sparse_categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     metrics=['accuracy'],
    # )

    # model.fit(
    #     train_ds,
    #     epochs=6,
    #     validation_data=val_ds,
    # )

    # model = get_and_compile_model(ConvClassifier)
    # loss, val_loss, acc, val_acc = supervised_train_loop(model, train, val, 32, 20)

    # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training History')

    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(loss, label='loss')
    # axes[0].plot(val_loss, label='val loss')
    # axes[0].legend()

    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].set_xlabel("Epoch", fontsize=14)
    # axes[1].plot(acc, label='acc')
    # axes[1].plot(val_acc, label='val acc')
    # axes[1].legend()
    # plt.show()
