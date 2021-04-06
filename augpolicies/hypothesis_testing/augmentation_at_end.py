from augpolicies.core.util import set_memory_growth
from augpolicies.core.classification import get_classificaiton_data, data_generator, get_and_compile_model, SimpleModel, ConvModel
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom, \
    apply_random_x_skew, apply_random_y_skew, apply_random_x_zoom, apply_random_y_zoom, apply_random_rotate, apply_random_cutout

import random
import numpy as np
import tensorflow as tf

import time
import csv

from augpolicies.core.util.parse_args import get_dataset_from_args
dataset = get_dataset_from_args()

file_name = "data/results/aug_at_end_data.csv"

aug_choices = [
    apply_random_left_right_flip,
    # apply_random_up_down_flip,
    apply_random_contrast,
    apply_random_skew,
    apply_random_zoom,
    # apply_random_x_skew,
    # apply_random_y_skew,
    # apply_random_x_zoom,
    # apply_random_y_zoom,
    apply_random_brightness,
    apply_random_rotate,
    # apply_random_cutout,
]

models = [SimpleModel, ConvModel]  # [SimpleModel, ConvModel]
batch_size = 256

e = 40
e_augs = list(range(0, e + 1, 10))

try:
    with open(file_name, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "policy_name", "aug", "model", "prob", "mag", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time"])
except FileExistsError:
    pass

names = ['interval', 'start', 'end']
policies = [{'interval': True}, {'start': True}, {'start': False}]

prob = 1.0
mag = 0.1

for _ in range(3):  # repeats
    for aug in aug_choices:
        for m in models:
            for n, p_kwargs in zip(names, policies):
                for e_aug in e_augs:
                    print(f"{aug.__name__} - {m.__name__} - {n} - {e_aug}")

                    _mag = mag
                    train, val, test = get_classificaiton_data(dataset=dataset)
                    model = get_and_compile_model(m)
                    t1 = time.time()

                    if aug is apply_random_left_right_flip or aug is apply_random_up_down_flip:
                        _mag = 1.0
                        func = [kwargs_func_prob(prob)]
                    else:
                        func = [kwargs_func_prob_mag(do_prob_mean=prob, mag_mean=_mag)]

                    p = HalfAugmentationPolicy([aug], func, e, e_aug, num_to_apply=1, **p_kwargs)
                    losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=p, batch_size=batch_size)
                    print(f'Time: {time.time() - t1:.2f}s')
                    with open(file_name, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_acc_idx = np.argmax(val_accs)
                        writer.writerow([dataset.__name__.split(".")[-1], n, aug.__name__, m.__name__, f"{prob}", f"{_mag}",
                                         f"{e}", f"{e_aug}",
                                         f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                                         f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                                         f"{time.time() - t1:.2f}"])
