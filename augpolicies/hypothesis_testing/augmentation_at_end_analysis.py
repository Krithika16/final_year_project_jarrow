import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aug_at_end_data.csv")

df = pd.pivot_table(df, values=['val_acc'], index=['name', 'e_augs'])

plt.plot(df.loc['end'])
plt.plot(df.loc['start'])
plt.plot(df.loc['interval'])

plt.legend(['end', 'start', 'interval'])
plt.show()
