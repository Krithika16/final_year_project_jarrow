from augpolicies.core.mnist import get_mnist, data_generator, get_and_compile_model, SimpleModel
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy

import random
import numpy as np


if __name__ == "__main__":
    from augpolicies.core.util import set_memory_growth
    import time
    train, val, test = get_mnist()
    model = get_and_compile_model(SimpleModel)

    def select_args():
        probs_11 = [p / 10 for p in range(11)]
        probs_4 = [0.0, 0.1, 0.25, 0.5]
        probs_3 = [0.0, 0.1, 0.2]
        mags_7 = [p / 10 for p in range(7)]
        # mags_shear = [p * 5 for p in range(5)]
        # mags_zoom = [p * 0.5 for p in range(5)]
        return [
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_4),),
            (random.choice(probs_4),),
            # (random.choice(probs_3), random.choice(mags_shear)),
            # (random.choice(probs_3), random.choice(mags_zoom)),
        ]

    def select_args():
        return [
            (0.1, 0.1),
            (0.1, 0.1),
            (0.5,),
            (0.5,),
        ]

    e = 20
    e_augs = list(range(e + 1))

    import csv
    with open("aug_at_end_data.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["name", "e", "e_augs", "loss", "val_loss", "acc", "val_acc", "time"])

    names = ['interval', 'start', 'end']
    policies = [{'interval': True}, {'start': True}, {'start': False}]
    for _ in range(2):  # repeats
        for n, p_kwargs in zip(names, policies):
            for e_aug in e_augs:
                t1 = time.time()
                p = HalfAugmentationPolicy(select_args, e, e_aug, **p_kwargs)
                losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=p)
                print(f'Time: {time.time() - t1:.2f}s')
                with open("aug_at_end_data.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    best_acc_idx = np.argmax(val_accs)
                    writer.writerow([n, f"{e}", f"{e_aug}",
                                    f"{losses[best_acc_idx]}", f"{val_losses[best_acc_idx]}",
                                    f"{accs[best_acc_idx]}", f"{val_accs[best_acc_idx]}",
                                    f"{time.time() - t1:.2f}"])
