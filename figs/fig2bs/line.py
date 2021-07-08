import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("test.txt", sep='\t')

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect(100)

k = 'ari'

for key, group in df.groupby('repeat'):

    group.sort_values('samples', inplace=True)
    print(key, group)
    ax.plot(group['samples'], group[k], color='red', linewidth=2, linestyle=':') 
    ax.fill_between(group['samples'], group[k]-group[k+'_std'], group[k]+group[k+'_std'], facecolor='red', alpha=0.2)


plt.xlabel('Number of samples')
plt.ylabel(k)
ax.set_xticks([15,30,45,60,75])
ax.set_xticklabels(('15', '30', '45', '60', '75'), fontsize=13)


plt.axis([11,79,0.60,1.01])
plt.savefig("testest.png")
