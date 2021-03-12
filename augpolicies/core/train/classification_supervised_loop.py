import time
import tensorflow as tf
import numpy as np


def get_train_step_fn():
    @tf.function
    def train_step(model, inputs, targets, optimizer, loss_func):
        with tf.GradientTape() as tape:
            pred = model(inputs, training=True)
            loss = loss_func(targets, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, pred
    return train_step


@tf.function
def val_step(model, inputs, targets, loss_func):
    pred = model(inputs, training=False)
    loss = loss_func(targets, pred)
    return loss, pred


def eval_loop(val_ds, model, loss=None):
    e_val_loss_avg = tf.keras.metrics.Mean()
    e_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    if loss is None:
        loss = model.loss
    for x, y in val_ds:
        val_loss, val_pred = val_step(model, x, y, loss)
        e_val_loss_avg.update_state(val_loss)
        e_val_acc.update_state(y, val_pred)
    return e_val_loss_avg.result(), e_val_acc.result()


def epoch(train_ds, val_ds, model, augmentation_policy, epoch_number, train_step_fn,
          loss=None, optimizer=None):
    e_loss_avg = tf.keras.metrics.Mean()
    e_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    if optimizer is None:
        optimizer = model.optimizer
    if loss is None:
        loss = model.loss
    for x, y in train_ds:
        if augmentation_policy is not None:
            x, y = augmentation_policy((x, y, epoch_number))
        tr_loss, tr_pred = train_step_fn(model, x, y, optimizer, loss)
        e_loss_avg.update_state(tr_loss)
        e_acc.update_state(y, tr_pred)
    e_val_loss_avg, e_val_acc = eval_loop(val_ds, model)
    return e_loss_avg.result(), e_val_loss_avg, e_acc.result(), e_val_acc


def supervised_train_loop(model, train, val, data_generator, augmentation_policy=None,
                          batch_size=128, epochs=20, debug=True, early_stop=None,
                          loss=None, optimizer=None):
    train_loss_results = []
    train_val_loss_results = []
    train_acc_results = []
    train_val_acc_results = []

    train_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*train, batch_size, True))
    val_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*val, batch_size, False))

    train_step_fn = get_train_step_fn()

    best_loss = np.inf
    best_loss_at = -1

    t0 = time.time()
    for e in range(epochs):

        e_loss_avg, e_val_loss_avg, e_acc, e_val_acc = epoch(train_ds, val_ds, model, augmentation_policy, e, train_step_fn, loss=loss, optimizer=optimizer)

        train_loss_results.append(e_loss_avg)
        train_val_loss_results.append(e_val_loss_avg)
        train_acc_results.append(e_acc)
        train_val_acc_results.append(e_val_acc)
        if debug:
            # tf.print(f"{e+1:03d}/{epochs:03d}: Loss: {train_loss_results[-1]:.3f}, Val Loss: {train_val_loss_results[-1]:.3f}, Acc: {train_acc_results[-1]:.3f}, Val Acc: {train_val_acc_results[-1]:.3f}")
            pass
        if early_stop:
            if e_val_loss_avg < best_loss:
                best_loss = e_val_loss_avg
                best_loss_at = e
            if e - best_loss_at >= early_stop:
                if debug:
                    print("Early stopping at:", e + 1)
                    break
    if debug:
        loss, acc = eval_loop(train_ds, model, loss=loss)
        tf.print(f"No Aug Loss: {loss:.3f}, No Aug Acc: {acc:.3f}, Duration: {time.time() - t0:.1f}, E: {e + 1}")
    return train_loss_results, train_val_loss_results, train_acc_results, train_val_acc_results
