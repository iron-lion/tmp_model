import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("tcga_lr_curv.txt", sep='\t')
print(df)

fig, ax = plt.subplots()
ax.axis([0,10100,0.0,1.01])
ax.set_aspect(5000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
my_color = {
    'Na:3':'blue',
    'TF:3': 'red',
}

for key, group in df.groupby('Dclass'):
    print(key)
    axs = group.plot('epochs', 'acc',
                #yerr='acc_std', 
               label=key, ax=ax,
               linestyle='-',
               color=my_color[key])
    axs.fill_between(group['epochs'], group['acc']-group['acc_std'], group['acc']+group['acc_std'],
                    facecolor=my_color[key], alpha=0.3)

plt.ylabel('Accuracy', fontsize=13)
#plt.legend(bbox_to_anchor=(1.00, 0), loc='lower right', fontsize='x-small')
ax.get_legend().remove()

plt.savefig("tcga_lr_curv_naive_tf.png")
