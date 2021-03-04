import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/results/aug_comparison.csv")
df = df.sort_values('val_acc')

pt = pd.pivot_table(df, values=['val_acc'], index=['aug', 'mag'])

args = ["apply_random_skew"]
for i in df['aug']:
    if i not in args:
        sns.scatterplot(x=pt.loc[i].index, y=pt.loc[i]['val_acc'], label=i)
        plt.yscale("log")
        args.append(i)
plt.show()
