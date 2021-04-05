import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/results/aug_comparison.csv")

from augpolicies.core.util.parse_args import get_dataset_from_args
dataset = get_dataset_from_args()
try:
    df = df[df['dataset' == dataset.__name__]]
except KeyError:
    pass

df = df.sort_values('val_acc')

pt = pd.pivot_table(df, values=['val_acc'], index=['aug', 'model', 'mag'])
pt_var = pd.pivot_table(df, values=['val_acc'], index=['aug', 'model', 'mag'], aggfunc=np.std)


excl_args = []

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = 13

for ax in [ax1, ax2]:
    ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

for m in df.model.unique():

    ax = ax1 if m == "SimpleModel" else ax2

    for i in df.aug.unique():
        if i not in excl_args:
            err = None
            try:
                err = pt_var.loc[i, m].val_acc
                if len(err) != len(pt.loc[i, m]['val_acc']):
                    raise KeyError
                ax.errorbar(x=pt.loc[i, m].index, y=pt.loc[i, m]['val_acc'], yerr=err, label=f'{i}', marker='o')
            except KeyError:
                ax.plot(pt.loc[i, m].index, pt.loc[i, m]['val_acc'], label=f'{i}', marker='o')


ax1.title.set_text('Simple Model')
ax2.title.set_text('Conv Model')

plt.tight_layout()
plt.legend()
plt.savefig("aug_comparison")
plt.show()
