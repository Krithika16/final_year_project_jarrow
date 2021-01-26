from augpolicies.core.mnist import get_mnist, data_generator, get_and_compile_model, SimpleModel
from augpolicies.core.train.classification_supervised_loop import supervised_train_loop
from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy

import random


if __name__ == "__main__":
    from augpolicies.core.util import set_memory_growth
    import time
    train, val, test = get_mnist()
    model = get_and_compile_model(SimpleModel)

    e = 10
    e_aug = 9

    def select_args():
        probs_11 = [p / 10 for p in range(11)]
        probs_4 = [0.0, 0.1, 0.25, 0.5]
        probs_3 = [0.0, 0.1, 0.2]
        mags_7 = [p/10 for p in range(7)]
        # mags_shear = [p * 5 for p in range(5)]
        # mags_zoom = [p * 0.5 for p in range(5)]
        return [
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_11), random.choice(mags_7)),
            (random.choice(probs_4),),
            (random.choice(probs_4),),
            #(random.choice(probs_3), random.choice(mags_shear)),
            #(random.choice(probs_3), random.choice(mags_zoom)),
        ]

    t1 = time.time()
    fixed = HalfAugmentationPolicy(select_args)
    losses, val_losses, accs, val_accs = supervised_train_loop(model, train, test, data_generator, epochs=e, augmentation_policy=fixed)

    print(f'Time: {time.time() - t1:.2f}s')


# list of augmentation functions


# classification task


