import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/results/aug_at_end_data.csv")

policies = df.policy_name.unique()
augs = df.aug_name.unique()
models = df.model.unique()

pt = pd.pivot_table(df, values=['val_acc'], index=['policy_name', 'aug_name', 'model', 'e_augs'])

f, ax = plt.subplots(nrows=len(models), ncols=len(augs), sharey='all')

for a_idx, a in enumerate(augs):
    for m_idx, m in enumerate(models):
        for p in policies:
            a_ = None
            if len(augs) == 1:
                a_ = ax[m_idx]
            elif len(models) == 1:
                a_ = ax[a_idx]
            else:
                a_ = ax[m_idx, a_idx]
            a_.plot(pt.loc[p, a, m])
            if m_idx == 0:
                a_.title.set_text(a)
            if a_idx == (len(augs) - 1):
                a_.yaxis.set_label_position("right")
                a_.set_ylabel(m)

plt.legend(['end', 'start', 'interval'])
plt.savefig("aug_at_end")
plt.tight_layout()
plt.show()
