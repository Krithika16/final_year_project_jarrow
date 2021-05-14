import tensorflow as tf
import os


def set_tf_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(tf.config.get_visible_devices())
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def set_tf_memory_growth_for_system():
    system_name = os.getenv("HOST_HOSTNAME")
    if system_name.upper() == "ARCHER":
        set_tf_memory_growth()
    elif system_name.upper() == "POMPEII-20":
        pass
    elif system_name.upper() == "":
        raise SystemError("Are you running inside a docker container with the host_hostname environment set?")
    else:
        raise NotImplementedError(f"Unknown system host name {system_name}")
