import pytest

def temp():
    return

@pytest.mark.parametrize(
    "e_total, aug_apps, expected_output",
    [
        (5, 0, []),
        (5, 1, [4]),
        (5, 2, [2, 4]),
        (5, 3, [1, 2, 4]),
        (5, 4, [0, 2, 3, 4]),
        (5, 5, [0, 1, 2, 3, 4]),
        (10, 10, [0,1,2,3,4,5,6,7,8,9]),
        (10,  9, [  1,2,3,4,5,6,7,8,9]),
        (10,  8, [  1,  3,4,5,6,7,8,9]),
        (10,  7, [  1,  3,  5,6,7,8,9]),
        (10,  6, [  1,  3,  5,  7,8,9]),
        (10,  5, [  1,  3,  5,  7,  9]),
        (10,  4, [    2,  4,    7,  9]),
        (10,  3, [    2,      6,    9]),
        (10,  2, [        4,        9]),
        (10,  1, [                  9]),
        (10,  0, [])
    ]
)
def test_half_aug_policy_idxes(e_total, aug_apps, expected_output):
    from augpolicies.augmentation_policies.baselines import HalfAugmentationPolicy

    pol = HalfAugmentationPolicy(temp, e_total, aug_apps, interval=True)

    idxes = pol.get_interval_idxes(e_total, aug_apps)
    print(idxes, "==", expected_output)
    assert idxes == expected_output
