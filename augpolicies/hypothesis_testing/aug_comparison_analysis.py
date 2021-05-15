import pandas as pd
import os; os.environ['MPLCONFIGDIR'] = "/tmp"
import matplotlib.pyplot as plt
import numpy as np
from augpolicies.core.util.parse_args import get_dataset_from_args


def main(args):
    file_path = "data/results/aug_comparison/"

    df = pd.read_csv(os.path.join(file_path, "aug_comparison.csv"))

    dataset = get_dataset_from_args()
    try:
        df = df[df['dataset' == dataset.__name__]]
    except KeyError:
        pass

    plot_prob = False
    if plot_prob:
        pt_idx = ['aug', 'model', 'prob', 'mag']
    else:
        pt_idx = ['aug', 'model', 'mag']
    xaxis = 'mag'
    assert xaxis in ['aug', 'mag']


    df = df.sort_values('val_acc')
    pt = pd.pivot_table(df, values=['val_acc'], index=pt_idx)
    pt_var = pd.pivot_table(df, values=['val_acc'], index=pt_idx, aggfunc=np.std)

    models = df.model.unique()

    excl_augs = []

    f, ax = plt.subplots(1, len(models), sharey=True)

    cm = plt.get_cmap('nipy_spectral')
    if plot_prob:
        NUM_COLORS = 0
        ms = [0] * len(models)
        for m_idx, m in enumerate(models):
            for aug in df.aug.unique():
                for prob in df.prob.unique():
                    try:
                        x = pt.loc[aug, m, prob].index
                        print(aug, m, prob)
                        ms[m_idx] += 1
                    except KeyError:
                        pass
        NUM_COLORS = max(ms)
    else:
        NUM_COLORS = len(df.aug.unique())

    for m_idx, m in enumerate(df.model.unique()):

        if len(models) > 1:
            ax_ = ax[m_idx]
        else:
            ax_ = ax

        ax_.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

        for aug in df.aug.unique():
            if aug not in excl_augs:
                for prob_idx, prob in enumerate(df.prob.unique()):
                    visual = False
                    if plot_prob:
                        visual = True
                        lbl = f'{aug}_{prob}'
                    else:
                        if prob_idx == 0:
                            visual = True
                        lbl = f'{aug}'

                    if visual:
                        err = None
                        try:
                            if plot_prob:
                                err = pt_var.loc[aug, m, prob].val_acc
                                y = pt.loc[aug, m, prob]['val_acc']
                                if xaxis == 'mag':
                                    x = pt.loc[aug, m, prob].index
                                elif xaxis == 'aug':
                                    x = [aug] * len(pt.loc[aug, m, prob]['val_acc'])
                            else:
                                err = pt_var.loc[aug, m].val_acc
                                y = pt.loc[aug, m]['val_acc']
                                if xaxis == 'mag':
                                    x = pt.loc[aug, m].index
                                elif xaxis == 'aug':
                                    x = [aug] * len(pt.loc[aug, m]['val_acc'])
                            if len(err) != len(y):
                                raise KeyError
                            ax_.errorbar(x=x, y=y, yerr=err, label=lbl, marker='o', capsize=5, elinewidth=0.3, markeredgewidth=1)
                        except KeyError:
                            try:
                                if plot_prob:
                                    if xaxis == 'mag':
                                        x = pt.loc[aug, m, prob].index
                                    elif xaxis == 'aug':
                                        x = [aug] * len(pt.loc[aug, m, prob]['val_acc'])
                                    y = pt.loc[aug, m, prob]['val_acc']
                                else:
                                    if xaxis == 'mag':
                                        x = pt.loc[aug, m].index
                                    elif xaxis == 'aug':
                                        x = [aug] * len(pt.loc[aug, m]['val_acc'])
                                    y = pt.loc[aug, m]['val_acc']
                                ax_.plot(x, y, label=lbl, marker='o')
                            except KeyError:
                                pass


    for m_idx, m in enumerate(df.model.unique()):
        if len(models) > 1:
            ax_ = ax[m_idx]
        else:
            ax_ = ax
        ax_.title.set_text(m)


    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, wspace=0.08, left=0.05)
    plt.savefig(os.path.join(file_path, "aug_comparison"))
    plt.show()
