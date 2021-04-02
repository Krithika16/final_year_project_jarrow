from augpolicies.core.util import set_memory_growth
from augpolicies.core.mnist import get_mnist, data_generator, get_and_compile_model, SimpleModel, ConvModel
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom, \
    apply_random_x_skew, apply_random_y_skew, apply_random_x_zoom, apply_random_y_zoom, apply_random_rotate, apply_random_cutout

import random
import numpy as np

import time
import csv

file_name = "data/results/aug_at_end_data.csv"


aug_choices = [
    # apply_random_left_right_flip,
    apply_random_up_down_flip,
    apply_random_contrast,
    apply_random_skew,
    # apply_random_zoom,
    # apply_random_x_skew,
    # apply_random_y_skew,
    # apply_random_x_zoom,
    # apply_random_y_zoom,
    # apply_random_brightness,
    # apply_random_rotate,
    # apply_random_cutout,
]


e = 3
e_augs = list(range(e + 1))

with open(file_name, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["policy_name", "aug_name", "model", "prob", "mag", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time"])

names = ['interval', 'start', 'end']
policies = [{'interval': True}, {'start': True}, {'start': False}]

prob = 0.5
mag = 0.1


models = [SimpleModel, ConvModel]

for aug in aug_choices:
    for m in models:
        for _ in range(1):  # repeats
            for n, p_kwargs in zip(names, policies):
                for e_aug in e_augs:
                    _mag = mag
                    train, val, test = get_mnist()
                    model = get_and_compile_model(m)
                    t1 = time.time()

                    if aug is apply_random_left_right_flip or aug is apply_random_up_down_flip:
                        _mag = 1.0
                        func = [kwargs_func_prob(prob)]
                    else:
                        func = [kwargs_func_prob_mag(do_prob_mean=prob, mag_mean=_mag)]

                    p = HalfAugmentationPolicy([aug], func, e, e_aug, num_to_apply=1, **p_kwargs)
                    losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=p)
                    print(f'Time: {time.time() - t1:.2f}s')
                    with open(file_name, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_acc_idx = np.argmax(val_accs)
                        writer.writerow([n, aug.__name__, m.__name__, f"{prob}", f"{_mag}",
                                         f"{e}", f"{e_aug}",
                                         f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                                         f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                                         f"{time.time() - t1:.2f}"])
