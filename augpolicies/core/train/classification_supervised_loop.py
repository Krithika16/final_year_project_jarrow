from augpolicies.core.util.reshape import make_3d, pad_to_min_size
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


def eval_loop(val_ds, model, loss):
    e_val_loss_avg = tf.keras.metrics.Mean()
    e_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in val_ds:
        val_loss, val_pred = val_step(model, x, y, loss)
        e_val_loss_avg.update_state(val_loss)
        e_val_acc.update_state(y, val_pred)
    return e_val_loss_avg.result(), e_val_acc.result()


def epoch(train_ds, val_ds, model, augmentation_policy, epoch_number, train_step_fn,
          loss, optimizer):
    e_loss_avg = tf.keras.metrics.Mean()
    e_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in train_ds:
        if augmentation_policy is not None:
            x, y = augmentation_policy((x, y, epoch_number))
        tr_loss, tr_pred = train_step_fn(model, x, y, optimizer, loss)
        e_loss_avg.update_state(tr_loss)
        e_acc.update_state(y, tr_pred)
    e_val_loss_avg, e_val_acc = eval_loop(val_ds, model, loss)
    return e_loss_avg.result(), e_val_loss_avg, e_acc.result(), e_val_acc


def get_lr_decay_closure(total_epochs: int, e_decay: int, *,
                         lr_decay_factor: float, lr_start: float, lr_min: float,
                         lr_warmup: float, warmup_proportion: float):
    # total_epochs: expected total training length
    # e_decay: number of epochs till decay
    # lr decay factor
    # lr at start after the warm up
    # min lr
    # lr during warmup, lr_warmup -> lr_start

    warmup_epoch_length = int(total_epochs * warmup_proportion)

    def lr_func(current_epoch, best_loss_at, learning_rate):
        updated_learning_rate = learning_rate
        if (current_epoch <= warmup_epoch_length - 1) and (warmup_epoch_length > 0):
            # warmup here
            warmup_left = (warmup_epoch_length - 1 - current_epoch) / (warmup_epoch_length)
            updated_learning_rate = lr_warmup * (warmup_left) + lr_start * (1 - warmup_left)
        else:
            # main loop with lr decay
            if current_epoch - best_loss_at >= e_decay:
                if (current_epoch - best_loss_at) % e_decay == 0:
                    temp_learning_rate = learning_rate * lr_decay_factor
                    if temp_learning_rate >= lr_min:
                        updated_learning_rate = temp_learning_rate
            else:
                updated_learning_rate = learning_rate
        return updated_learning_rate
    return lr_func


def supervised_train_loop(model, train, val, data_generator, *, augmentation_policy=None,
                          batch_size=128, epochs=20, debug=True,
                          early_stop=None, lr_decay=None,
                          loss=None, optimizer=None):
    train_loss_results = []
    train_val_loss_results = []
    train_acc_results = []
    train_val_acc_results = []

    if model.requires_3d:
        train = make_3d(train)
        val = make_3d(val)

    train = pad_to_min_size(train, model.min_size)
    val = pad_to_min_size(val, model.min_size)

    train_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*train, batch_size, True))
    val_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*val, batch_size, False))

    train_step_fn = get_train_step_fn()

    best_loss = np.inf
    best_loss_at = -1

    if optimizer is None:
        optimizer = model.optimizer
    if loss is None:
        loss = model.loss

    t0 = time.time()
    for e in range(epochs):

        if lr_decay:
            optimizer.lr = lr_decay(e, best_loss_at, optimizer.lr)

        e_loss_avg, e_val_loss_avg, e_acc, e_val_acc = epoch(train_ds, val_ds, model, augmentation_policy, e, train_step_fn, loss=loss, optimizer=optimizer)

        train_loss_results.append(e_loss_avg)
        train_val_loss_results.append(e_val_loss_avg)
        train_acc_results.append(e_acc)
        train_val_acc_results.append(e_val_acc)

        if e_val_loss_avg < best_loss:
            best_loss = e_val_loss_avg
            best_loss_at = e
        if debug:
            print(f"{e+1:03d}/{epochs:03d}: Loss: {train_loss_results[-1]:.3f},",
                  f"Val Loss: {train_val_loss_results[-1]:.3f}, Acc: {train_acc_results[-1]:.3f},",
                  f"Val Acc: {train_val_acc_results[-1]:.3f}, Time so far: {time.time() - t0:.1f}, Lr: {optimizer.lr.numpy():.5f}, Since Best: {e - best_loss_at}")
        if early_stop:
            if e - best_loss_at >= early_stop:
                if debug:
                    print("Early stopping at:", e + 1)
                    break
    if debug:
        eval_loss, eval_acc = eval_loop(train_ds, model, loss=loss)
        tf.print(f"No Aug Loss: {eval_loss:.3f}, No Aug Acc: {eval_acc:.3f}, Duration: {time.time() - t0:.1f}, E: {e + 1}")
    return train_loss_results, train_val_loss_results, train_acc_results, train_val_acc_results
