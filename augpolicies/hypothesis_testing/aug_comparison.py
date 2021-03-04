from augpolicies.core.mnist import get_mnist, data_generator, get_and_compile_model, SimpleModel
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import NoAugmentationPolicy, AugmentationPolicy
from augpolicies.augmentation_funcs.augmentation_2d import apply_random_brightness, \
    apply_random_contrast, apply_random_left_right_flip, apply_random_up_down_flip, apply_random_skew, apply_random_zoom
from augpolicies.augmentation_funcs.augmentation_2d import kwargs_func_prob, kwargs_func_prob_mag

import random
import numpy as np
import time
import csv

with open("aug_comparison.csv", 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["aug", "e", "early_stop_e", "prob", "mag", "loss", "val_loss", "acc", "val_acc", "time"])


if __name__ == "__main__":
    from augpolicies.core.util import set_memory_growth

    train, val, test = get_mnist()
    t1 = time.time()

    e = 50

    aug_choices = [
        apply_random_left_right_flip,
        apply_random_up_down_flip,
        apply_random_contrast,
        apply_random_skew,
        apply_random_zoom,
        apply_random_brightness,
    ]

    for i in range(5):
        t1 = time.time()
        ap = NoAugmentationPolicy()
        model = get_and_compile_model(SimpleModel)
        losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=ap)
        with open("aug_comparison.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            best_acc_idx = np.argmax(val_accs)
            writer.writerow(["No Aug", f"{e}", f"{best_acc_idx+1}", "", "",
                             f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                             f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                             f"{time.time() - t1:.2f}"])

    for idx, aug in enumerate(aug_choices):
        aug = [aug_choices[idx]]
        prob = 0.5
        mag = 0.0

        if idx < 2:
            for i in range(5):
                t1 = time.time()
                func = [kwargs_func_prob(prob)]
                ap = AugmentationPolicy(aug, func, num_to_apply=1)
                model = get_and_compile_model(SimpleModel)
                losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=ap)
                with open("aug_comparison.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    best_acc_idx = np.argmax(val_accs)
                    writer.writerow([aug[0].__name__, f"{e}", f"{best_acc_idx+1}", f"{prob}", "",
                                     f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                                     f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                                     f"{time.time() - t1:.2f}"])
        else:
            for i in range(10):
                t1 = time.time()
                func = [kwargs_func_prob_mag(do_prob_mean=prob, mag_mean=mag + 0.15 * i)]
                ap = AugmentationPolicy(aug, func, num_to_apply=1)
                model = get_and_compile_model(SimpleModel)
                losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=ap)
                with open("aug_comparison.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    best_acc_idx = np.argmax(val_accs)
                    writer.writerow([aug[0].__name__, f"{e}", f"{best_acc_idx+1}", f"{prob}", f"{mag}",
                                     f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                                     f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                                     f"{time.time() - t1:.2f}"])
