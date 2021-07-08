import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("naive_15_baron_2nd.txt", sep='\t')
df = df[df.Dclass != 'Baron']
#print(df)

df2 = pd.read_csv("tf_15_baron.txt", sep='\t')
df2 = df2[df2.Dclass != 'Baron_TF']
#print(df2)


fig, ax = plt.subplots()
ax.axis([0,201000,0.2,0.91])
ax.set_aspect(150000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
my_color = {
    'Muraro':'blue',
    'Seger': 'goldenrod',
    'Wang' : 'green',
    'Xin' : 'red',
    'Muraro_TF':'blue',
    'Seger_TF': 'goldenrod',
    'Wang_TF' : 'green',
    'Xin_TF' : 'red',
}

for key, group in df2.groupby('Dclass'):
    axs = group.plot('epochs', 'acc',
                #yerr='acc_std', 
               label=key, ax=ax, color=my_color[key])
    axs.fill_between(group['epochs'], group['acc']-group['acc_std'], group['acc']+group['acc_std'],
                    facecolor=my_color[key], alpha=0.3)

for key, group in df.groupby('Dclass'):
    axs = group.plot('epochs', 'acc',
                #yerr='acc_std', 
               label=key, ax=ax,
               linestyle=':',
               color=my_color[key])
    axs.fill_between(group['epochs'], group['acc']-group['acc_std'], group['acc']+group['acc_std'],
                    facecolor=my_color[key], alpha=0.15)

plt.axvline(x=100000, linewidth=1, color='black', linestyle='--')
plt.ylabel('Accuracy', fontsize=13)
#plt.legend(bbox_to_anchor=(1.00, 0), loc='lower right', fontsize='x-small')
ax.get_legend().remove()

plt.savefig("line_lr_curv_baron.png")
