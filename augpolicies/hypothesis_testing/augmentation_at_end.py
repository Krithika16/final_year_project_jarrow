from augpolicies.core.util import set_memory_growth
from augpolicies.core.classification import (get_classificaiton_data, data_generator,
                                             get_and_compile_model)
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop, get_lr_decay_closure
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_left_right_flip, apply_random_up_down_flip

import time
import random
import csv
import json
import os

from augpolicies.core.util.parse_args import get_dataset_from_args, get_config_json
dataset = get_dataset_from_args()
config = get_config_json()

results_path = "data/results/aug_at_end/"
file_name = "aug_at_end"
file_path = os.path.join(results_path, f"{file_name}.csv")

try:
    os.makedirs(results_path)
except FileExistsError:
    pass

e_augs = list(range(0, config['epochs'] + 1, 2))
lr_decay = get_lr_decay_closure(config['epochs'], config['lr']['decay'],
                                lr_decay_factor=config['lr']['decay_factor'],
                                lr_start=config['lr']['start'], lr_min=config['lr']['min'],
                                lr_warmup=config['lr']['warmup'],
                                warmup_proportion=config['lr']['warmup_prop'])

try:
    with open(file_path, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "policy_name", "aug", "model", "prob", "mag", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time", "results_tag"])
except FileExistsError:
    pass

names = ['interval', 'start', 'end']
policies = [{'interval': True}, {'start': True}, {'start': False}]

mag = config['aug']['mag']
prob = config['aug']['prob']

for _ in range(config['repeats']):  # repeats
    for aug in config['aug']['choices']:
        for m in config['models']:
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

                    p = HalfAugmentationPolicy([aug], func, config['epochs'], e_aug,
                                               num_to_apply=config['aug']['num_to_apply'], **p_kwargs)
                    with open(file_path) as f:
                        num_lines = sum(1 for line in f)
                    id_tag = f"{file_name}_{num_lines + 1}"
                    h = supervised_train_loop(model, train, test, data_generator, id_tag=id_tag,
                                              epochs=config['epochs'], augmentation_policy=p,
                                              batch_size=config['batch_size'], lr_decay=lr_decay)
                    print(f'Time: {time.time() - t1:.2f}s')
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_idx = h['best_val_loss']['epoch']
                        writer.writerow([dataset.__name__.split(".")[-1], n, aug.__name__, m.__name__, f"{prob}", f"{_mag}",
                                         f"{config['epochs']}", f"{e_aug}",
                                         f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                         f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                         f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                    with open(os.path.join(results_path, f"{h['file_name']}.json"), "w") as f:
                        json.dump(h, f)
