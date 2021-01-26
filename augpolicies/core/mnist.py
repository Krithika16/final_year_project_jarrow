from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import \
    NoAugmentationPolicy, FixAugmentationPolicy, RandomAugmentationPolicy
import tensorflow as tf
import numpy as np
import random


def get_mnist(dataset=tf.keras.datasets.fashion_mnist.load_data, val_split=None, normalise_factor=255.0):
    (x_train, y_train), (x_test, y_test) = dataset()
    x_train = x_train[..., np.newaxis] / normalise_factor
    x_test = x_test[..., np.newaxis] / normalise_factor
    if val_split is not None:
        val_length = int(len(x_train) * val_split)
        x_train, y_train = x_train[:-val_length, ...], y_train[:-val_length, ...]
        x_val, y_val = x_train[-val_length:, ...], y_train[-val_length:, ...]
        x_val = np.float32(x_val)
    else:
        x_val = y_val = None

    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    train = (x_train, y_train)
    val = (x_val, y_val)
    test = (x_test, y_test)
    return train, val, test


def data_generator(x, y, batch_size=32, train=True):
    x_len = len(x)
    idxes = np.array(list(range(x_len)))
    if train:
        np.random.shuffle(idxes)
    idx = 0
    while idx + batch_size < (x_len - 1):
        batch_idxes = idxes[idx: idx + batch_size]
        x_ = x[batch_idxes]
        y_ = y[batch_idxes]
        yield (x_, y_)
        idx += batch_size


class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.model_layers = [
            tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10)
        ]

    def call(self, x, training=False):
        for lay in self.model_layers:
            x = lay(x)
        return x


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ]

    def call(self, x, training=False):
        for lay in self.model_layers:
            x = lay(x)
        return x


def get_and_compile_model(model_func, lr=0.001):
    model = model_func()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


if __name__ == "__main__":
    from augpolicies.core.util import set_memory_growth
    import time
    train, val, test = get_mnist()
    model = get_and_compile_model(ConvModel)

    e = 5

    def select_args():
        probs_11 = [p/10 for p in range(11)]
        probs_4 = [0.0, 0.1, 0.25, 0.5]
        probs_3 = [0.0, 0.1, 0.2]
        mags_7 = [p/10 for p in range(7)]
        # mags_shear = [p * 5 for p in range(5)]
        # mags_zoom = [p * 0.5 for p in range(5)]
        return [
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_4),),
            (random.choice(probs_4),),
            #(random.choice(probs_3), random.choice(mags_shear)),
            #(random.choice(probs_3), random.choice(mags_zoom)),
        ]

    args = []
    last_val_accs = []
    for i in range(4):
        print("EXPERIMENT:", i)
        t1 = time.time()
        if i < 2:
            losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=NoAugmentationPolicy())
            args.append(None)
        elif i < 4:
            fixed = FixAugmentationPolicy(select_args)
            losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=fixed)
            args.append(fixed.aug_args)
        else:
            rnd = RandomAugmentationPolicy(select_args)
            losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=rnd)
            args.append("rnd policy")
        print(f'Time: {time.time() - t1:.2f}s')
        last_val_accs.append(val_accs[-1])

    acc_args = list(zip(last_val_accs, args))
    acc_args.sort(key=lambda x: x[0])
    for acc, ag in acc_args:
        print(acc.numpy(), ag)

# todo: determine the magnitudes and probabilities that is optimal for random and fixed augmentation policy
