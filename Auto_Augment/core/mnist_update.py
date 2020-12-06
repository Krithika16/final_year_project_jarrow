from Auto_Augment.core.util import set_memory_growth
import tensorflow as tf
import tensorflow_datasets as tfds

def get_data(batch_size=32):
    (ds_train, ds_test) = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test

if __name__=='__main__':
    get_data()