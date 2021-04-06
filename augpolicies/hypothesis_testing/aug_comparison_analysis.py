import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from augpolicies.core.util.parse_args import get_dataset_from_args

df = pd.read_csv("data/results/aug_comparison.csv")

dataset = get_dataset_from_args()
try:
    df = df[df['dataset' == dataset.__name__]]
except KeyError:
    pass

df = df.sort_values('val_acc')

pt = pd.pivot_table(df, values=['val_acc'], index=['aug', 'model', 'mag'])
pt_var = pd.pivot_table(df, values=['val_acc'], index=['aug', 'model', 'mag'], aggfunc=np.std)


models = df.model.unique()

excl_augs = []

f, ax = plt.subplots(1, len(models), sharey=True)


cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = len(df.aug.unique())

for m_idx, m in enumerate(df.model.unique()):

    if len(models) > 1:
        ax_ = ax[m_idx]
    else:
        ax_ = ax

    ax_.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    for aug in df.aug.unique():
        if aug not in excl_augs:
            err = None
            try:
                err = pt_var.loc[aug, m].val_acc
                if len(err) != len(pt.loc[aug, m]['val_acc']):
                    raise KeyError
                ax_.errorbar(x=pt.loc[aug, m].index, y=pt.loc[aug, m]['val_acc'], yerr=err, label=f'{aug}', marker='o')
            except KeyError:
                try:
                    ax_.plot(pt.loc[aug, m].index, pt.loc[aug, m]['val_acc'], label=f'{aug}', marker='o')
                except KeyError:
                    pass


for m_idx, m in enumerate(df.model.unique()):
    if len(models) > 1:
        ax_ = ax[m_idx]
    else:
        ax_ = ax
    ax_.title.set_text(m)

plt.tight_layout()
plt.legend()
plt.savefig("aug_comparison")
plt.show()
