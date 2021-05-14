from augpolicies.core.util.reshape import make_3d, pad_to_min_size
import time
import tensorflow as tf
import numpy as np
from datetime import datetime


def get_train_step_fn(strategy):
    with strategy.scope():
        @tf.function
        def train_step(model, inputs, targets, optimizer, loss_func):
            with tf.GradientTape() as tape:
                pred = model(inputs, training=True)
                loss = loss_func(targets, pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, pred
    return train_step

def get_val_step_fn(strategy):
    with strategy.scope():
        @tf.function
        def val_step(model, inputs, targets, loss_func):
            pred = model(inputs, training=False)
            loss = loss_func(targets, pred)
            return loss, pred
    return val_step


def eval_loop(val_ds, model, loss, val_step):
    e_val_loss_avg = tf.keras.metrics.Mean()
    e_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in val_ds:
        val_loss, val_pred = val_step(model, x, y, loss)
        e_val_loss_avg.update_state(val_loss)
        e_val_acc.update_state(y, val_pred)
    return e_val_loss_avg.result(), e_val_acc.result()

def get_epoch_fn(strategy):
    with strategy.scope():
        def epoch(train_ds, val_ds, model, augmentation_policy, epoch_number,
                train_step_fn, val_step_fn,
                loss, optimizer):
            e_loss_avg = tf.keras.metrics.Mean()
            e_acc = tf.keras.metrics.SparseCategoricalAccuracy()
            for x, y in train_ds:
                if augmentation_policy is not None:
                    x, y = augmentation_policy((x, y, epoch_number))
                tr_loss, tr_pred = train_step_fn(model, x, y, optimizer, loss)
                e_loss_avg.update_state(tr_loss)
                e_acc.update_state(y, tr_pred)
            e_val_loss_avg, e_val_acc = eval_loop(val_ds, model, loss, val_step_fn)
            return e_loss_avg.result(), e_val_loss_avg, e_acc.result(), e_val_acc
    return epoch

class get_lr_decay_closure:
    def __init__(self, total_epochs: int, e_decay: int, *,
                 lr_decay_factor: float, lr_start: float, lr_min: float,
                 lr_warmup: float, warmup_proportion: float):
        # total_epochs: expected total training length
        # e_decay: number of epochs till decay
        # lr decay factor
        # lr at start after the warm up
        # min lr
        # lr during warmup, lr_warmup -> lr_start
        self.total_epochs = total_epochs
        self.e_decay = e_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_warmup = lr_warmup
        self.warmup_proportion = warmup_proportion
        self.warmup_epoch_length = int(total_epochs * warmup_proportion)
        self.config = {
            'total_epochs': total_epochs, 'e_decay': e_decay, 'lr_decay_factor': lr_decay_factor,
            'lr_start': lr_start, 'lr_min': lr_min, 'lr_warmup': lr_warmup, 'warmup_proportion': warmup_proportion,
            'warmup_epoch_length': self.warmup_epoch_length,
        }

    def __call__(self, current_epoch, best_loss_at, learning_rate):
        updated_learning_rate = learning_rate.numpy()
        if (current_epoch <= self.warmup_epoch_length - 1) and (self.warmup_epoch_length > 0):
            # warmup here
            warmup_left = (self.warmup_epoch_length - 1 - current_epoch) / (self.warmup_epoch_length)
            updated_learning_rate = self.lr_warmup * (warmup_left) + self.lr_start * (1 - warmup_left)
        else:
            # main loop with lr decay
            if current_epoch - best_loss_at >= self.e_decay:
                if (current_epoch - best_loss_at) % self.e_decay == 0:
                    temp_learning_rate = updated_learning_rate * self.lr_decay_factor
                    if temp_learning_rate >= self.lr_min:
                        updated_learning_rate = temp_learning_rate
        return updated_learning_rate


def supervised_train_loop(model, train, val, data_generator, id_tag, strategy, *, 
                          augmentation_policy=None,
                          batch_size=128, epochs=20, debug=True,
                          early_stop=None, lr_decay=None,
                          loss=None, optimizer=None):

    history = {}
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

    # strategy functions
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = strategy.experimental_distribute_dataset(val_ds)
    train_step_fn = get_train_step_fn(strategy)
    val_step_fn = get_val_step_fn(strategy)
    epoch = get_epoch_fn(strategy)

    best_loss = np.inf
    best_loss_at = -1

    if optimizer is None:
        optimizer = model.optimizer
    if loss is None:
        loss = model.loss

    t0 = time.time()
    for e in range(epochs):

        if lr_decay:
            lr = lr_decay(e, best_loss_at, optimizer.learning_rate)
            optimizer.learning_rate = lr

        e_loss_avg, e_val_loss_avg, e_acc, e_val_acc = epoch(train_ds, val_ds, model, augmentation_policy, e,
                                                             train_step_fn, val_step_fn,
                                                             loss=loss, optimizer=optimizer)

        train_loss_results.append(e_loss_avg.numpy().tolist())
        train_val_loss_results.append(e_val_loss_avg.numpy().tolist())
        train_acc_results.append(e_acc.numpy().tolist())
        train_val_acc_results.append(e_val_acc.numpy().tolist())

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

    history['train_losses'] = train_loss_results
    history['val_losses'] = train_val_loss_results
    history['train_acc'] = train_acc_results
    history['val_acc'] = train_val_acc_results
    history['best_val_loss'] = {'loss': best_loss.numpy().item(),
                                'epoch': best_loss_at}
    history['train_eval_loss'] = eval_loss.numpy().item()
    history['train_eval_acc'] = eval_acc.numpy().item()
    history['target_epochs'] = epochs
    history['epochs_ran'] = e
    history['train_time'] = time.time() - t0
    history['file_name'] = f"{id_tag}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"

    history['strategy'] = strategy
    history['loss'] = loss.name
    history['optimizer'] = str(optimizer)
    history['early_stop'] = early_stop if early_stop else "NA"
    history['batch_size'] = batch_size
    history['lr_decay'] = lr_decay.config if lr_decay else "NA"
    history['aug_policy'] = augmentation_policy.config if augmentation_policy else "NA"

    return history
