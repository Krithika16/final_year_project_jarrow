from Auto_Augment.core.util.supervised_loop import supervised_train_loop
import tensorflow as tf
import numpy as np



def get_mnist(dataset=tf.keras.datasets.mnist.load_data, val_split=0.01):
    (x_train, y_train), (x_test, y_test) = dataset()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    val_length = int(len(x_train) * val_split)
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
    if train:
        np.random.shuffle(idxes)
    idx = 0
    while idx + batch_size < (x_len - 1):
        batch_idxes = idxes[idx: idx+batch_size]
        x_ = x[batch_idxes]
        y_ = y[batch_idxes]
        yield (x_, y_)
        idx += batch_size


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model_layers = [
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

    def call(self, x, training=False):
        for lay in self.model_layers:
            x = lay(x)
        return x


def get_and_compile_model(model_func, lr=0.001):
    model = model_func()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=['accuracy'],
    )
    return model


if __name__ == "__main__":
    from Auto_Augment.core.util import set_memory_growth
    train, val, test = get_mnist()
    model = get_and_compile_model(SimpleModel)
    supervised_train_loop(model, train, test, data_generator)
