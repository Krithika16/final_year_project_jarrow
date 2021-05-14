import time
from datetime import datetime
import csv
import json
import os

from augpolicies.core.util.system_hardware import set_tf_memory_growth_for_system
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop, get_lr_decay_closure
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_left_right_flip, apply_random_up_down_flip, apply_no_aug
from augpolicies.augmentation_policies.baselines import AugmentationPolicy, NoAugmentationPolicy
from augpolicies.core.util.parse_args import get_dataset_from_args, get_config_json


set_tf_memory_growth_for_system()
dataset = get_dataset_from_args()
config = get_config_json()

start_time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
task = os.path.splitext(os.path.basename(__file__))[0]

results_path = f"data/results/{task}/{start_time}/"
results_file = os.path.join(results_path, "summary_results.csv")

try:
    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, "episode"))
except FileExistsError:
    pass

with open(os.path.join(results_path, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

try:
    with open(results_file, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "aug", "model", "e", "early_stop_e", "prob", "mag", "loss", "val_loss", "acc", "val_acc", "time", "results_tag"])
except FileExistsError:
    pass

lr_decay = get_lr_decay_closure(config['epochs'], config['lr']['decay'],
                                lr_decay_factor=config['lr']['decay_factor'],
                                lr_start=config['lr']['start'], lr_min=config['lr']['min'],
                                lr_warmup=config['lr']['warmup'],
                                warmup_proportion=config['lr']['warmup_prop'])

for _ in range(config['repeats']):
    for i in range(4):
        for m in config['models']:
            t1 = time.time()
            p = NoAugmentationPolicy()
            with open(results_file) as f:
                num_lines = sum(1 for line in f)
            id_tag = f"{task}_{num_lines + 1}"
            h = supervised_train_loop(m, dataset, id_tag=id_tag,
                                      strategy=config['strategy'], epochs=config['epochs'],
                                      augmentation_policy=p, batch_size=config['batch_size'],
                                      lr_decay=lr_decay)
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                best_idx = h['best_val_loss']['epoch']
                writer.writerow([dataset.__name__.split(".")[-1], "No Aug", f"{m.__name__}",
                                 f"{config['epochs']}", f"{best_idx+1}", "-0.1", "-0.1",
                                 f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                 f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                 f"{time.time() - t1:.2f}", f"{h['file_name']}"])
            with open(os.path.join(results_path, "episode", f"{h['file_name']}.json"), "w") as f:
                json.dump(h, f, indent=4)

    for idx, aug in enumerate(config['aug']['choices']):
        prob = 0.0
        mag = 0.0

        if (aug is apply_random_left_right_flip) or (aug is apply_random_up_down_flip) or (aug is apply_no_aug):
            for prob_f in range(4):
                for m in config['models']:
                    _prob = 0.25 * (prob_f + 1)
                    _mag = 1.0
                    t1 = time.time()
                    func = [kwargs_func_prob(_prob)]
                    p = AugmentationPolicy([aug], func, num_to_apply=1)
                    with open(results_file) as f:
                        num_lines = sum(1 for line in f)
                    id_tag = f"{task}_{num_lines + 1}"
                    h = supervised_train_loop(m, dataset, id_tag=id_tag,
                                              strategy=config['strategy'], epochs=config['epochs'],
                                              augmentation_policy=p, batch_size=config['batch_size'],
                                              lr_decay=lr_decay)
                    with open(results_file, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_idx = h['best_val_loss']['epoch']
                        writer.writerow([dataset.__name__.split(".")[-1], aug.__name__, f"{m.__name__}",
                                         f"{config['epochs']}", f"{best_idx+1}", f"{_prob}", f"{_mag}",
                                         f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                         f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                         f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                    with open(os.path.join(results_path, "episode", f"{h['file_name']}.json"), "w") as f:
                        json.dump(h, f, indent=4)
        else:
            for mag_f in range(5):
                for prob_f in range(2):
                    for m in config['models']:
                        aug_ = aug
                        _mag = 0.0 + (0.25 * mag_f)
                        _prob = 0.5 + (0.5 * prob_f)
                        t1 = time.time()
                        func = [kwargs_func_prob_mag(do_prob_mean=_prob, mag_mean=_mag)]
                        p = AugmentationPolicy([aug_], func, num_to_apply=1)
                        with open(results_file) as f:
                            num_lines = sum(1 for line in f)
                        id_tag = f"{task}_{num_lines + 1}"
                        h = supervised_train_loop(m, dataset, id_tag=id_tag,
                                                  strategy=config['strategy'], epochs=config['epochs'],
                                                  augmentation_policy=p, batch_size=config['batch_size'],
                                                  lr_decay=lr_decay)
                        with open(results_file, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            best_idx = h['best_val_loss']['epoch']
                            writer.writerow([dataset.__name__.split(".")[-1], aug.__name__, f"{m.__name__}",
                                             f"{config['epochs']}", f"{best_idx+1}", f"{_prob}", f"{_mag}",
                                             f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                             f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                             f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                        with open(os.path.join(results_path, "episode", f"{h['file_name']}.json"), "w") as f:
                            json.dump(h, f, indent=4)
