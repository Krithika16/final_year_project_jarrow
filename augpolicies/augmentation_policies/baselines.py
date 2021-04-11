import tensorflow as tf
import tensorflow_addons as tfa
import random
import math
from typing import Tuple, List, Callable


class NoAugmentationPolicy(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(NoAugmentationPolicy, self).__init__()
        self.config = "NA"

    def call(
        self,
        inputs: Tuple[tfa.types.TensorLike, tfa.types.TensorLike, int],
        training: bool = False
    ) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

        x, y, e = inputs
        return x, y


class AugmentationPolicy(tf.keras.Model):
    def __init__(
        self,
        aug_choices: List[Callable],
        aug_kwargs_funcs_list: List[Callable],
        apply_to_y: bool = False,
        num_to_apply: int = 2
    ):

        super(AugmentationPolicy, self).__init__()
        if aug_kwargs_funcs_list is None:
            aug_kwargs_funcs_list = [None] * len(aug_choices)
        assert len(aug_choices) == len(aug_kwargs_funcs_list)
        assert num_to_apply <= len(aug_choices)
        self.num_to_apply = num_to_apply
        self.aug_choices = aug_choices
        self.aug_kwargs_funcs_list = aug_kwargs_funcs_list
        self.config = {
            'num_to_apply': num_to_apply,
            'aug_choices': [i.__name__ for i in aug_choices],
        }

    def call(
        self,
        inputs: Tuple[tfa.types.TensorLike, tfa.types.TensorLike, int],
        training: bool = False
    ) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

        x, y, e = inputs
        aug_idxes = random.sample(range(len(self.aug_choices)), self.num_to_apply)
        for idx in aug_idxes:
            aug = self.aug_choices[idx]
            kwargs = self.aug_kwargs_funcs_list[idx]()
            if kwargs is None:
                kwargs = {}
            x, y = aug(x, y, **kwargs)
        return x, y


class HalfAugmentationPolicy(tf.keras.Model):
    def __init__(
        self,
        aug_choices: List[Callable],
        aug_kwargs_funcs_list: List[Callable],
        e_total: int,
        aug_applications: int,
        apply_to_y: bool = False,
        start: bool = None,
        interval: bool = None,
        num_to_apply: int = 2
    ) -> None:

        super(HalfAugmentationPolicy, self).__init__()
        if aug_kwargs_funcs_list is None:
            aug_kwargs_funcs_list = [None] * len(aug_choices)
        assert len(aug_choices) == len(aug_kwargs_funcs_list)
        assert num_to_apply <= len(aug_choices)
        assert start is not None or interval is not None
        if start is None:
            assert interval is not None
        if interval is None:
            assert start is not None
        assert e_total >= aug_applications >= 0
        self.start = start
        self.interval = interval
        self.num_to_apply = num_to_apply
        self.aug_kwargs_funcs_list = aug_kwargs_funcs_list
        self.aug_applications = aug_applications
        self.e_total = e_total

        if start:
            self.apply_aug_func = self.split_start
        elif start is not None:
            self.apply_aug_func = self.split_end
        elif interval:
            self.idxes = self.get_interval_idxes(e_total, aug_applications)
            self.apply_aug_func = self.split_interval
        self.aug_choices = aug_choices
        self.config = {
            'start': start,
            'interval': interval,
            'num_to_apply': num_to_apply,
            'aug_choices': [i.__name__ for i in aug_choices],
            'aug_applications': aug_applications,
        }

    def call(
        self,
        inputs: Tuple[tfa.types.TensorLike, tfa.types.TensorLike, int],
        training: bool = False
    ) -> Tuple[tfa.types.TensorLike, tfa.types.TensorLike]:

        x, y, e = inputs
        apply_aug = self.apply_aug_func(e)
        if apply_aug:
            aug_idxes = random.sample(range(len(self.aug_choices)), self.num_to_apply)
            for idx in aug_idxes:
                aug = self.aug_choices[idx]
                kwargs = self.aug_kwargs_funcs_list[idx]()
                if kwargs is None:
                    kwargs = {}
                x, y = aug(x, y, **kwargs)
        return x, y

    def split_start(self, e: int) -> bool:
        return e < self.aug_applications

    def split_end(self, e: int) -> bool:
        return e >= (self.e_total - self.aug_applications)

    def split_interval(self, e: int) -> bool:
        return e in self.idxes

    def get_interval_idxes(
        self,
        e_total: int,
        aug_applications: int
    ) -> List[int]:
        is_even = e_total % 2 == 0
        split_idx = None
        if is_even:
            split_idx = (e_total // 2) + 1
        else:
            split_idx = math.ceil(e_total / 2) + 1
        above_half = aug_applications >= split_idx
        idxes = []
        if aug_applications == 0:
            return idxes
        if above_half:
            idx = e_total
            while (idx > 0) and len(idxes) < aug_applications:
                idxes.append(idx - 1)
                idx -= 2
            idx = e_total - 1
            while len(idxes) < aug_applications:
                idxes.append(idx - 1)
                idx -= 2
        else:
            idx_freq = e_total / aug_applications
            idx = e_total
            prev_target = target = idx
            while (idx > 0) and len(idxes) < aug_applications:
                if idxes == []:
                    idxes.append(idx - 1)
                    prev_target = target
                    target = prev_target - idx_freq
                else:
                    if abs(idx - target) <= 0.5:
                        prev_target = target
                        target = prev_target - idx_freq
                        idxes.append(idx - 1)
                idx -= 1
        idxes = sorted(idxes)
        assert len(idxes) == aug_applications
        assert e_total - 1 in idxes
        return idxes


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def temp():
        return

    e_total = 5
    aug_apps = 2
    expected_output = [2, 4]

    pol = HalfAugmentationPolicy(temp, e_total, aug_apps, interval=True)
    idxes = pol.idxes
    print(idxes)
    print(expected_output)
    assert idxes == expected_output
