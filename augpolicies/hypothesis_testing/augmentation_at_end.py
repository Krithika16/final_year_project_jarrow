from augpolicies.core.util import set_memory_growth
from augpolicies.core.classification import (get_classificaiton_data, data_generator,
                                             get_and_compile_model,
                                             SimpleModel, ConvModel, EfficientNetB0)
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop, get_lr_decay_closure
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom, \
    apply_random_x_skew, apply_random_y_skew, apply_random_x_zoom, apply_random_y_zoom, apply_random_rotate, apply_random_cutout

import numpy as np
import tensorflow as tf

import time
import random
import csv
import json
import os

from augpolicies.core.util.parse_args import get_dataset_from_args
dataset = get_dataset_from_args()

results_path = "data/results/aug_at_end/"
file_name = "aug_at_end_data_skew_12"
file_path = os.path.join(results_path, f"{file_name}.csv")

try:
    os.makedirs(results_path)
except FileExistsError:
    pass

e = 4
e_augs = list(range(0, e + 1, 2))
batch_size = 256

lr_decay = 4
lr_decay_factor = 0.5
lr_warmup = 1e-5
lr_start = 3e-3
lr_min = 1e-5
lr_warmup_prop = 0.1


lr_decay = get_lr_decay_closure(e, lr_decay, lr_decay_factor=lr_decay_factor,
                                lr_start=lr_start, lr_min=lr_min,
                                lr_warmup=lr_warmup, warmup_proportion=lr_warmup_prop)

aug_choices = [
    # apply_random_left_right_flip,
    apply_random_up_down_flip,
    # apply_random_contrast,
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

models = [EfficientNetB0]  # SimpleModel, ConvModel, EfficientNetB0

try:
    with open(file_path, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "policy_name", "aug", "model", "prob", "mag", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time", "results_tag"])
except FileExistsError:
    pass

names = ['interval', 'start', 'end']
policies = [{'interval': True}, {'start': True}, {'start': False}]

prob = 1.0
mag = 0.2
repeats = 10

for _ in range(repeats):  # repeats
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
                    with open(file_path) as f:
                        num_lines = sum(1 for line in f)
                    id_tag = f"{file_name}_{num_lines + 1}"
                    h = supervised_train_loop(model, train, test, data_generator, id_tag=id_tag, epochs=e, augmentation_policy=p, batch_size=batch_size, lr_decay=lr_decay)
                    print(f'Time: {time.time() - t1:.2f}s')
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_idx = h['best_val_loss']['epoch']
                        writer.writerow([dataset.__name__.split(".")[-1], n, aug.__name__, m.__name__, f"{prob}", f"{_mag}",
                                         f"{e}", f"{e_aug}",
                                         f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                         f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                         f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                    with open(os.path.join(results_path, f"{h['file_name']}.json"), "w") as f:
                        json.dump(h, f)
