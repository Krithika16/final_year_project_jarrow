import tensorflow as tf


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


def eval_loop(val_ds, model):
    e_val_loss_avg = tf.keras.metrics.Mean()
    e_val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    loss = model.loss
    for x, y in val_ds:
        val_loss, val_pred = val_step(model, x, y, loss)
        e_val_loss_avg.update_state(val_loss)
        e_val_acc.update_state(y, val_pred)
    return e_val_loss_avg.result(), e_val_acc.result()


def epoch(train_ds, val_ds, model, augmentation_policy, epoch_number):
    e_loss_avg = tf.keras.metrics.Mean()
    e_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = model.optimizer
    loss = model.loss
    for x, y in train_ds:
        if augmentation_policy is not None:
            x, y = augmentation_policy((x, y, epoch_number))
        tr_loss, tr_pred = train_step(model, x, y, optimizer, loss)
        e_loss_avg.update_state(tr_loss)
        e_acc.update_state(y, tr_pred)
    e_val_loss_avg, e_val_acc = eval_loop(val_ds, model)
    return e_loss_avg.result(), e_val_loss_avg, e_acc.result(), e_val_acc


def supervised_train_loop(model, train, val, data_generator, augmentation_policy=None,
                          batch_size=128, epochs=20, debug=True):
    train_loss_results = []
    train_val_loss_results = []
    train_acc_results = []
    train_val_acc_results = []

    train_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*train, batch_size, True))
    val_ds = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.int32), args=(*val, batch_size, False))

    for e in range(epochs):

        e_loss_avg, e_val_loss_avg, e_acc, e_val_acc = epoch(train_ds, val_ds, model, augmentation_policy, e)

        train_loss_results.append(e_loss_avg)
        train_val_loss_results.append(e_val_loss_avg)
        train_acc_results.append(e_acc)
        train_val_acc_results.append(e_val_acc)
        if debug:
            # tf.print(f"{e+1:03d}/{epochs:03d}: Loss: {train_loss_results[-1]:.3f}, Val Loss: {train_val_loss_results[-1]:.3f}, Acc: {train_acc_results[-1]:.3f}, Val Acc: {train_val_acc_results[-1]:.3f}")

            if e == (epochs - 1):
                loss, acc = eval_loop(train_ds, model)
                tf.print(f"No Aug Loss: {loss:.3f}, No Aug Acc: {acc:.3f}")
    return train_loss_results, train_val_loss_results, train_acc_results, train_val_acc_results
