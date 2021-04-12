import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from augpolicies.core.util.parse_args import get_dataset_from_args
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


file_path = "data/results/aug_comparison/"

df = pd.read_csv(os.path.join(file_path, "aug_comparison.csv"))

dataset = get_dataset_from_args()
try:
    df = df[df['dataset' == dataset.__name__]]
except KeyError:
    pass

val_losses = []

for idx, row in df.iterrows():
    current_json = os.path.join(file_path, f"{row['results_tag']}.json")
    with open(current_json, "r") as f:
        data = json.load(f)
    losses = data['val_losses']
    best_loss = data['best_val_loss']['loss']
    losses.append(best_loss)
    val_losses.append(losses)

val_losses = np.array(val_losses)
val_losses_ranked = np.zeros_like(val_losses)

for col_num in range(val_losses.shape[1]):
    ranked = rankdata(val_losses[:, col_num], method='min')
    val_losses_ranked[:, col_num] = ranked


def calculate_lwma_rank(ranks, f=1):
    lwma_num = 0
    divisor = 0
    for r2_idx, r2_item in enumerate(ranks):
        lwma_num += f * (r2_idx + 1) * r2_item
        divisor += f * (r2_idx + 1)
    lwma_num /= divisor
    return lwma_num


def calculate_ewma_rank(ranks, k=0.5):
    ewma_num = 0
    for item in ranks:
        ewma_num = (k * item) + (ewma_num * (1 - k))
    return ewma_num


def calculate_avg_rank(ranks):
    av_num = sum(ranks)/sum(range(len(ranks)+1))
    return av_num


def get_errors_for_data_proportion(data_to_skip, val_losses_ranked, fs=[1], ks=[0.5], threshold=0.1):
    val_losses_proxy_values = []
    names = None
    for idx, r2 in enumerate(val_losses_ranked):
        true_rank = r2[-1]
        ranks = r2[:-1-data_to_skip]
        last_rank = ranks[-1]
        row = [idx, true_rank, last_rank]
        if idx == 0:
            names = ["idx", "true", "last"]
        row.append(calculate_avg_rank(ranks))
        for i in fs:
            row.append(calculate_lwma_rank(ranks, f=i))
            if idx == 0:
                names.append(f"lwma_{i:.2f}")
        for i in ks:
            row.append(calculate_ewma_rank(ranks, k=i))
            if idx == 0:
                names.append(f"ewma_{i:.2f}")
        val_losses_proxy_values.append(row)
        if idx == 0:
            names.append("avg")

    val_losses_proxy_values = np.array(val_losses_proxy_values)
    val_losses_proxy_ranks = np.zeros_like(val_losses_proxy_values)

    for col_num in range(val_losses_proxy_values.shape[1]):
        if col_num > 1:
            ranked = rankdata(val_losses_proxy_values[:, col_num], method='min')
            val_losses_proxy_ranks[:, col_num] = ranked
        else:
            val_losses_proxy_ranks[:, col_num] = val_losses_proxy_values[:, col_num]

    errors = []
    top_errors = []
    top_threshold = int(threshold * val_losses_proxy_ranks.shape[0]) - 1

    true_top_ranks = val_losses_proxy_ranks[:, 1]
    true_top_ranks[true_top_ranks > top_threshold] = top_threshold * 10
    for col_num in range(val_losses_proxy_ranks.shape[1]):
        mse = mean_squared_error(np.exp(-val_losses_proxy_ranks[:, 1]), np.exp(-val_losses_proxy_ranks[:, col_num]))
        # mse = mean_squared_error(val_losses_proxy_ranks[:, 1], val_losses_proxy_ranks[:, col_num])
        errors.append(mse)

        col_top_ranks = val_losses_proxy_ranks[:, col_num]
        col_top_ranks_threshold = np.sort(col_top_ranks)[top_threshold]
        col_top_ranks[col_top_ranks > col_top_ranks_threshold] = top_threshold * 10

        top_mse = mean_squared_error(np.exp(-true_top_ranks), np.exp(-col_top_ranks))
        # top_mse = mean_squared_error(true_top_ranks, col_top_ranks)
        top_errors.append(top_mse)
    return errors, top_errors, names


errors = []
top_errors = []
n = None
for i in reversed(range(val_losses_ranked.shape[1] - 1)):
    err, t10_err, n = get_errors_for_data_proportion(i, val_losses_ranked, fs=np.arange(0.7, 1.4, 0.1), ks=np.arange(0.1, 1.1, 0.1))
    errors.append(err)
    top_errors.append(t10_err)

errors = np.array(errors)

cm = plt.get_cmap('nipy_spectral')
NUM_COLORS = len(n)
plt.gca().set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

for col in range(errors.shape[1]):
    if col != 0:
        plt.plot(errors[:, col], label=n[col])
plt.title("Analysis of Estimation of Rank")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend()

plt.figure()

top_errors = np.array(top_errors)

cm = plt.get_cmap('nipy_spectral')
NUM_COLORS = len(n)
plt.gca().set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

for col in range(top_errors.shape[1]):
    if col != 0:
        plt.plot(top_errors[:, col], label=n[col])
plt.legend()
plt.title("Analysis of Estimation of Top Ranked")
plt.ylabel("MSE")
plt.xlabel("Epoch")

plt.show()
