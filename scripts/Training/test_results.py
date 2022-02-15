import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_html('output/df.html')[0]
print(df)

sns.scatterplot(x='volfrac', y = 'loss', data=df)
plt.savefig('output/test_accuracy.png')
plt.clf()

sns.scatterplot(x='noise', y = 'loss', data=df)
plt.savefig('output/test_accuracy_noise.png')
plt.clf()

sns.scatterplot(x='n_particles', y = 'loss', data=df)
plt.savefig('output/test_accuracy_n_particles.png')
plt.clf()

sns.scatterplot(x='z_gauss', y = 'loss', data=df)
plt.savefig('output/test_accuracy_z_gauss.png')
plt.clf()

sns.scatterplot(x='r', y = 'loss', data=df)
plt.savefig('output/test_accuracy_r.png')
plt.clf()