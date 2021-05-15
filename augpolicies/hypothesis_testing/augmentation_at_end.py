import time
from datetime import datetime
import csv
import json
import os

from augpolicies.core.util.system_hardware import set_tf_memory_growth_for_system
from augpolicies.core.util.dict2json import serializable_objects_in_dict
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop, get_lr_decay_closure
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_left_right_flip, apply_random_up_down_flip
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy
from augpolicies.core.util.parse_args import get_args


set_tf_memory_growth_for_system()
args = get_args()
dataset = args.config
config = args.dataset

task = os.path.splitext(os.path.basename(__file__))[0]

results_path = f"data/results/{task}/{config['log_id']}/"
results_file = os.path.join(results_path, "summary_results.csv")

try:
    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, "episode"))
except FileExistsError:
    pass

serializable_config = serializable_objects_in_dict(config)
with open(os.path.join(results_path, "config.json"), "w") as f:
    json.dump(serializable_config, f, indent=4)

try:
    with open(results_file, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["dataset", "policy_name", "aug", "model", "prob", "mag", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time", "results_tag"])
except FileExistsError:
    pass

e_augs = list(range(0, config['epochs'] + 1, 2))
lr_decay = get_lr_decay_closure(config['epochs'], config['lr']['decay'],
                                lr_decay_factor=config['lr']['decay_factor'],
                                lr_start=config['lr']['start'], lr_min=config['lr']['min'],
                                lr_warmup=config['lr']['warmup'],
                                warmup_proportion=config['lr']['warmup_prop'])

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
                    t1 = time.time()

                    if aug is apply_random_left_right_flip or aug is apply_random_up_down_flip:
                        _mag = 1.0
                        func = [kwargs_func_prob(prob)]
                    else:
                        func = [kwargs_func_prob_mag(do_prob_mean=prob, mag_mean=_mag)]

                    p = HalfAugmentationPolicy([aug], func, config['epochs'], e_aug,
                                               num_to_apply=config['aug']['num_to_apply'], **p_kwargs)
                    with open(results_file) as f:
                        num_lines = sum(1 for line in f)
                    id_tag = f"{task}_{num_lines + 1}"
                    h = supervised_train_loop(m, dataset, id_tag=id_tag,
                                              strategy=config['strategy'], epochs=config['epochs'],
                                              augmentation_policy=p, batch_size=config['batch_size'],
                                              lr_decay=lr_decay)
                    print(f'Time: {time.time() - t1:.2f}s')
                    with open(results_file, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        best_idx = h['best_val_loss']['epoch']
                        writer.writerow([dataset.__name__.split(".")[-1], n, aug.__name__, m.__name__, f"{prob}", f"{_mag}",
                                         f"{config['epochs']}", f"{e_aug}",
                                         f"{h['train_losses'][best_idx]}", f"{h['val_losses'][best_idx]}",
                                         f"{h['train_acc'][best_idx]}", f"{h['val_acc'][best_idx]}",
                                         f"{time.time() - t1:.2f}", f"{h['file_name']}"])
                    with open(os.path.join(results_path, "episode", f"{h['file_name']}.json"), "w") as f:
                        json.dump(h, f, indent=4)
