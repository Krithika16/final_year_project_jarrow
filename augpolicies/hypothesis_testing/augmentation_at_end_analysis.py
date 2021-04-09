import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("data/results/aug_at_end_data_skew_20.csv")

from augpolicies.core.util.parse_args import get_dataset_from_args
dataset = get_dataset_from_args()
try:
    df = df[df['dataset' == dataset.__name__]]
except KeyError:
    pass

policies = df.policy_name.unique()

augs = df.aug.unique()
models = df.model.unique()

pt = pd.pivot_table(df, values=['val_acc'], index=['policy_name', 'aug', 'model', 'e_augs'])
pt_var = pd.pivot_table(df, values=['val_acc'], index=['policy_name', 'aug', 'model', 'e_augs'], aggfunc=np.std)

f, ax = plt.subplots(nrows=len(models), ncols=len(augs), sharey='all', sharex='all')

for a_idx, a in enumerate(augs):
    for m_idx, m in enumerate(models):
        for p in policies:
            a_ = None
            if len(augs) == 1 and len(models) == 1:
                a_ = ax
            elif len(augs) == 1:
                a_ = ax[m_idx]
            elif len(models) == 1:
                a_ = ax[a_idx]
            else:
                a_ = ax[m_idx, a_idx]
            try:
                x = pt.loc[p, a, m].index
                y = pt.loc[p, a, m].val_acc
                err = pt_var.loc[p, a, m].val_acc
                if len(y) == len(err):
                    a_.errorbar(x=x, y=y, yerr=err, label=f"{a}_{p}", marker='o', capsize=5, elinewidth=0.3, markeredgewidth=1)
                else:
                    raise KeyError
            except KeyError:
                try:
                    a_.plot(x, y, label=f"{a}_{p}")
                except KeyError:
                    pass

            if m_idx == 0:
                a_.title.set_text(a)
            if a_idx == (len(augs) - 1):
                a_.yaxis.set_label_position("right")
                a_.set_ylabel(m)
                a_.legend()

plt.tight_layout()
plt.savefig("aug_at_end")
plt.show()
