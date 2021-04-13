import time
import random
import csv
import json
import os
import numpy as np
import tensorflow as tf
from augpolicies.augmentation_funcs.augmentation_2d import (
    apply_no_aug, apply_random_brightness, apply_random_contrast, apply_random_cutout,
    apply_random_left_right_flip, apply_random_rotate, apply_random_skew,
    apply_random_up_down_flip, apply_random_x_skew, apply_random_x_zoom,
    apply_random_y_skew, apply_random_y_zoom, apply_random_zoom,
    kwargs_func_prob, kwargs_func_prob_mag)
from augpolicies.augmentation_policies.baselines import (AugmentationPolicy,
                                                         NoAugmentationPolicy)
from augpolicies.core.classification import (ConvModel, SimpleModel, EfficientNetB0,
                                             data_generator,
                                             get_and_compile_model,
                                             get_classificaiton_data)
from augpolicies.core.train.classification_supervised_loop import \
    supervised_train_loop, get_lr_decay_closure
from augpolicies.core.util import set_memory_growth

results_path = "data/results/aug_comparison/"
file_name = "aug_comparison"
file_path = os.path.join(results_path, f"{file_name}.csv")

try:
    os.makedirs(results_path)
except FileExistsError:
    pass

try:
    with open(file_path, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "aug", "model", "e", "early_stop_e", "prob", "mag", "loss", "val_loss", "acc", "val_acc", "time", "results_tag"])
except FileExistsError:
    pass


e = 60
estop = 10
batch_size = 256
repeat = 3

lr_decay = 4
lr_decay_factor = 0.5
lr_warmup = 1e-5
lr_start = 3e-3
lr_min = 1e-5
lr_warmup_prop = 0.1


lr_decay = get_lr_decay_closure(e, lr_decay, lr_decay_factor=lr_decay_factor,
                                lr_start=lr_start, lr_min=lr_min,
                                lr_warmup=lr_warmup, warmup_proportion=lr_warmup_prop)
# lr_decay = None
aug_choices = [
    # apply_no_aug,
    apply_random_left_right_flip,
    # apply_random_up_down_flip,
    apply_random_contrast,
    apply_random_skew,
    apply_random_zoom,
    # apply_random_x_skew,
    # apply_random_y_skew,
    # apply_random_x_zoom,
    # apply_random_y_zoom,
    # apply_random_brightness,
    apply_random_rotate,
    # apply_random_cutout,
]

models = [EfficientNetB0]

from augpolicies.core.util.parse_args import get_dataset_from_args
dataset = get_dataset_from_args()

for _ in range(repeat):
    for i in range(4):
        for m in models:
            t1 = time.time()
            ap = NoAugmentationPolicy()
            model = get_and_compile_model(m)
            train, val, test = get_classificaiton_data(dataset=dataset)
            with open(file_path) as f:
                num_lines = sum(1 for line in f)
            id_tag = f"{file_name}_{num_lines + 1}"
            h = supervised_train_loop(model, train, test, data_generator, id_tag=id_tag, epochs=e, augmentation_policy=ap,
                                      early_stop=estop, batch_size=batch_size, lr_decay=lr_decay)
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                best_idx = h['best_val_loss']['epoch']
                writer.writerow([dataset.__name__.split(".")[-1], "No Aug", f"{m.__name__}", f"{e}", f"{best_idx+1}", "-0.1", "-0.1",
                                 f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                 f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                 f"{time.time() - t1:.2f}", f"{h['file_name']}"])
            with open(os.path.join(results_path, f"{h['file_name']}.json"), "w") as f:
                json.dump(h, f)

    for idx, aug in enumerate(aug_choices):
        aug = aug_choices[idx]
        prob = 0.0
        mag = 0.0

        if (aug is apply_random_left_right_flip) or (aug is apply_random_up_down_flip) or (aug is apply_no_aug):
            for prob_f in range(4):
                for m in models:
                    _prob = 0.25 * (prob_f + 1)
                    _mag = 1.0
                    _prob = tf.constant(_prob)
                    _mag = tf.constant(_mag)
                    t1 = time.time()
                    func = [kwargs_func_prob(_prob)]
                    ap = AugmentationPolicy([aug], func, num_to_apply=1)
                    model = get_and_compile_model(m)
                    train, val, test = get_classificaiton_data(dataset=dataset)
                    with open(file_path) as f:
                        num_lines = sum(1 for line in f)
                    id_tag = f"{file_name}_{num_lines + 1}"
                    h = supervised_train_loop(model, train, test, data_generator, id_tag=id_tag, epochs=e, augmentation_policy=ap,
                                              early_stop=estop, batch_size=batch_size, lr_decay=lr_decay)
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_idx = h['best_val_loss']['epoch']
                        writer.writerow([dataset.__name__.split(".")[-1], aug.__name__, f"{m.__name__}", f"{e}", f"{best_idx+1}", f"{_prob}", f"{_mag}",
                                         f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                         f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                         f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                    with open(os.path.join(results_path, f"{h['file_name']}.json"), "w") as f:
                        json.dump(h, f)
        else:
            for mag_f in range(5):
                for prob_f in range(2):
                    for m in models:
                        aug_ = aug
                        _mag = 0.0 + (0.25 * mag_f)
                        _prob = 0.5 + (0.5 * prob_f)
                        _prob = tf.constant(_prob)
                        _mag = tf.constant(_mag)
                        t1 = time.time()
                        func = [kwargs_func_prob_mag(do_prob_mean=_prob, mag_mean=_mag)]
                        ap = AugmentationPolicy([aug_], func, num_to_apply=1)
                        model = get_and_compile_model(m)
                        train, val, test = get_classificaiton_data(dataset=dataset)
                        with open(file_path) as f:
                            num_lines = sum(1 for line in f)
                        id_tag = f"{file_name}_{num_lines + 1}"
                        h = supervised_train_loop(model, train, test, data_generator, id_tag=id_tag, epochs=e, augmentation_policy=ap,
                                                  early_stop=estop, batch_size=batch_size, lr_decay=lr_decay)
                        with open(file_path, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            best_idx = h['best_val_loss']['epoch']
                            writer.writerow([dataset.__name__.split(".")[-1], aug.__name__, f"{m.__name__}", f"{e}", f"{best_idx+1}", f"{_prob}", f"{_mag}",
                                             f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                             f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                             f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                        with open(os.path.join(results_path, f"{h['file_name']}.json"), "w") as f:
                            json.dump(h, f)
