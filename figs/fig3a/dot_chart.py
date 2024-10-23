import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dot_plot_data.txt", sep='\t')
df = df[df.Dclass != 'Baron']
#print(df)

plt.figure(num=1,figsize=(12,6), dpi=600)
fig, ax = plt.subplots()

ax.axis([0,37,0.2,1.01])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect(15)

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
my_marker = {
    'TF' : 'd',
    'Naive_15' : 's',
    'TF_15' : 's',
    'Naive_Full' : 'o',
    'TF_Full': 'o',
}
my_fill = {
    'TF' : True,
    'Naive_15' : False,
    'TF_15' : True,
    'Naive_Full' : False,
    'TF_Full': True,
}
for key, gr in df.groupby('Dclass'):
    for desc, group in gr.groupby('desc'):
        print(key, my_color[key])
        axs = group.plot.scatter('pos', 'acc', s=30,
                    yerr='acc_std', 
                   label=key, ax=ax,
                   color= my_color[key] if my_fill[desc] is True else 'none',
                   marker=my_marker[desc],
                   edgecolors=my_color[key],
                   linewidths=2.0)

plt.ylabel('Accuracy', fontsize=11)
ax.get_legend().remove()
ax.set_xticks([3,8,13,18,22,26,30,34])
ax.set_xticklabels(('Muraro', 'Xin', 'Seger', 'Wang','Muraro', 'Xin', 'Seger', 'Wang'), fontsize=11)

plt.savefig("dot_chart_comp_all.png")
