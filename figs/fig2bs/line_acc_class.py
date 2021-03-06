import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_tf_class.txt", sep='\t')
df2 = pd.read_csv("data_class.txt", sep='\t')

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect(12.3)

k = 'acc'

for key, group in df.groupby('repeat'):

    group.sort_values('class', inplace=True)
    print(key, group)
    ax.plot(group['class'], group[k], color='red', linewidth=0.7, linestyle=':') 
    ax.fill_between(group['class'], group[k]-group[k +'_std'], group[k]+group[k+'_std'], facecolor='red', alpha=0.2)


for key, group in df2.groupby('repeat'):

    group.sort_values('class', inplace=True)
    print(key, group)
    ax.plot(group['class'], group[k], color='blue', linewidth=0.7, linestyle=':') 
    ax.fill_between(group['class'], group[k]-group[k+'_std'], group[k]+group[k+'_std'], facecolor='blue', alpha=0.2)

ax.set_xticks([5,8,10,15,20,25])
ax.set_xticklabels(('5', '8', '10', '15', '20', '25'), fontsize=12)


plt.xlabel('Number of classes')
plt.ylabel('Accuracy', fontsize=14)

plt.axis([4,26,0.0,1.01])
plt.savefig("acc_line_classes.png")
