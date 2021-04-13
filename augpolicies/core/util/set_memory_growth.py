import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#     print(tf.config.get_visible_devices())
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, False)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
