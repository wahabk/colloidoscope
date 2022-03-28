import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_html('output/df.html')[0]
print(df)

fig, axs = plt.subplots(2,2)

# TODO make this plot P and R

sns.scatterplot(x='volfrac', y = 'loss', data=df, ax=axs[0,0])
sns.scatterplot(x='noise', y = 'loss', data=df, ax=axs[0,1])
sns.scatterplot(x='z_gauss', y = 'loss', data=df, ax=axs[1,0])
sns.scatterplot(x='r', y = 'loss', data=df, ax=axs[1,1])
fig.tight_layout()
plt.savefig('output/test_accuracy.png')