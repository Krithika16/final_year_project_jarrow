import time
from datetime import datetime
import tensorflow as tf
import numpy as np
from augpolicies.core.classification import get_and_compile_model, get_classification_data
from augpolicies.core.util.reshape import make_3d, pad_to_min_size


def get_train_step_fn(model, optimizer, loss_func, acc_metric):
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_func(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc_metric.update_state(y, y_pred)  # accuracy metric
        return loss
    return train_step


def get_distributed_train_step_fn(train_step):
    @tf.function
    def distributed_train_step(strategy, inputs):
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=-1)
    return distributed_train_step


def get_val_step_fn(model, loss_func, loss_metric, acc_metric):
    def val_step(inputs):
        x,y = inputs
        y_pred = model(x, training=False)
        loss = loss_func(y, y_pred)
        loss_metric.update_state(loss)
        acc_metric.update_state(y, y_pred)
    return val_step


def get_distributed_val_step_fn(val_step):
    @tf.function
    def distributed_val_step(strategy, inputs):
        return strategy.run(val_step, args=(inputs,))
    return distributed_val_step


def get_eval_loop_fn(distributed_val_step_fn,  val_loss_metric, val_acc_metric):
    def eval_loop(strategy, val_ds):
        for inputs in val_ds:
            distributed_val_step_fn(strategy, inputs)
        val_loss = val_loss_metric.result()
        val_acc = val_acc_metric.result()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()
        return val_loss, val_acc
    return eval_loop


def get_epoch_fn(distributed_train_step_fn, eval_loop_fn,
                 train_acc_metric,):
    def epoch(strategy, train_ds, val_ds,
              augmentation_policy, epoch_number):
        train_loss = 0.0
        for inputs in train_ds:
            breakpoint()
            print(len(inputs))
            if augmentation_policy is not None:
                inputs = augmentation_policy(*inputs, epoch_number)
            train_loss += distributed_train_step_fn(strategy, inputs)
        val_loss, val_acc = eval_loop_fn(strategy, val_ds)

        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()
        return train_loss, val_loss, train_acc, val_acc
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


def supervised_train_loop(model_template, dataset, id_tag, strategy, *,
                          get_data_func=get_classification_data,
                          augmentation_policy=None,
                          batch_size=128, epochs=20, debug=True,
                          early_stop=None, lr_decay=None,
                          loss=None, optimizer=None):

    history = {}
    train_loss_results = []
    train_val_loss_results = []
    train_acc_results = []
    train_val_acc_results = []

    with strategy.scope():
        model = get_and_compile_model(model_template, loss, optimizer)
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        train_acc_metric= tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_acc'
        )

    train, val, _ = get_data_func(dataset=dataset)
    if model.requires_3d:
        train = make_3d(train)
        val = make_3d(val)

    train = pad_to_min_size(train, model.min_size)
    val = pad_to_min_size(val, model.min_size)

    train_ds = tf.data.Dataset.from_tensor_slices(train)
    train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(val).batch(batch_size)

    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = strategy.experimental_distribute_dataset(val_ds)

    train_step = get_train_step_fn(model, model.optimizer, model.loss, train_acc_metric)
    val_step = get_val_step_fn(model, model.loss, val_loss_metric, val_acc_metric)
    dist_train_step = get_distributed_train_step_fn(train_step)
    dist_val_step = get_distributed_val_step_fn(val_step)
    eval_loop = get_eval_loop_fn(dist_val_step, val_loss_metric, val_acc_metric)

    epoch = get_epoch_fn(dist_train_step, eval_loop,
                         train_acc_metric)

    best_loss = np.inf
    best_loss_at = -1

    t0 = time.time()
    for e in range(epochs):

        if lr_decay:
            lr = lr_decay(e, best_loss_at, model.optimizer.learning_rate)
            model.optimizer.learning_rate = lr

        e_loss_avg, e_val_loss_avg, e_acc, e_val_acc = epoch(strategy, train_ds, val_ds,
                                                             augmentation_policy, e)

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
                  f"Val Acc: {train_val_acc_results[-1]:.3f}, Time so far: {time.time() - t0:.1f}, Lr: {model.optimizer.lr.numpy():.5f}, Since Best: {e - best_loss_at}")
        if early_stop:
            if e - best_loss_at >= early_stop:
                if debug:
                    print("Early stopping at:", e + 1)
                    break
    if debug:
        eval_loss, eval_acc = eval_loop(strategy, train_ds)
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

    history['strategy_str'] = str(strategy)
    history['loss'] = model.loss.name
    history['optimizer'] = str(model.optimizer)
    history['early_stop'] = early_stop if early_stop else "NA"
    history['batch_size'] = batch_size
    history['lr_decay'] = lr_decay.config if lr_decay else "NA"
    history['aug_policy'] = augmentation_policy.config if augmentation_policy else "NA"

    return history
