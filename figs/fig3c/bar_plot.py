import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

my_color = {
    'Baron' : 'black',
    'Muraro':'blue',
    'Seg': 'goldenrod',
    'Wang' : 'green',
    'Xin' : 'red',
    'Baron' : 'black',
    'Muraro_TF':'blue',
    'Seg_TF': 'goldenrod',
    'Wang_TF' : 'green',
    'Xin_TF' : 'red',
}

my_plot_order = {
    'Baron' : 0,
    'Muraro' : 1,
    'Seg' : 2,
    'Wang' : 3,
    'Xin' : 4,
}
my_order=['Baron', 'Muraro', 'Seg','Wang','Xin']
key_i = 0
plt.figure(figsize=(9,9), dpi=300)

acc_ari = 'acc'

def one_row(df, axs, titles, xaxis):
    for key, group in df.groupby('Dclass'):
        # chart shape
        ax = axs[my_plot_order[key]]
        ax.spines['left'].set_bounds((0,1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axis([-1,20,0.0,1.1])

        # data
        data1, data2 = group[acc_ari][:10], group[acc_ari][10:]

        # statistical significance
        _, pv = stats.ttest_ind(data1, data2)
        print(key, pv)
        ax.violinplot([data1, data2], widths=6, positions=[5, 15], showextrema=False, showmedians=True)
        y, h = group[acc_ari].max()+0.05, 0.08
        ax.plot([5,5,15,15], [data1.max()+0.05, y+h, y+h, data2.max()+0.05], lw=0.7, c='k')
        
        if (-math.log10(pv) > 3):
            aster = '***'
        elif (-math.log10(pv) > 2):
            aster = '**'
        elif (-math.log10(pv) > -math.log10(0.05)):
            aster = '*'
        else:
            aster = ''

        ax.text(10 , y+h+0.03, aster, ha='center', va='bottom')

        # chart outlook
        ax.get_xaxis().set_ticks([])
        if key != 'Baron':
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticks([])
        else:
            global key_i
            #ax.set_ylabel('ARI\n\n'+my_order[key_i])
            ax.set_ylabel('Accuracy\n', fontsize=13)
            key_i+=1
        if titles == True :
            ax.set_title(key, y=1.3, pad=1)
        if xaxis == True:
            ax.spines['bottom'].set_visible(True)
            ax.set_xticks([5,15])
            ax.set_xticklabels(('TF', 'Random'), fontsize=13)
            ax.spines['bottom'].set_bounds((3,18))


def override_w_text(texts, axs, yaxis=False, xaxis=False):
    axs.axis([-1,20,0.0,1.1])
    axs.text(10,0.5,texts, fontsize=14, horizontalalignment='center', verticalalignment='center')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.get_xaxis().set_ticks([])
    axs.spines['bottom'].set_visible(False)

    if yaxis:
        axs.spines['left'].set_bounds((0,1.0))
        axs.set_ylabel('Accuracy\n', fontsize=13)
    else:
        axs.spines['left'].set_visible(False)
        axs.get_yaxis().set_ticks([])

    if xaxis:
        axs.spines['bottom'].set_visible(True)
        axs.set_xticks([5,15])
        axs.set_xticklabels(('TF', 'Random'), fontsize=13)
        axs.spines['bottom'].set_bounds((3,18))



#### First
df = pd.read_csv("baron_value.txt", sep='\t')

ax0 = plt.subplot2grid((5,5), (0,0))
ax1 = plt.subplot2grid((5,5), (0,1))
ax2 = plt.subplot2grid((5,5), (0,2))
ax3 = plt.subplot2grid((5,5), (0,3))
ax4 = plt.subplot2grid((5,5), (0,4))
axs = [ax0, ax1, ax2, ax3, ax4]

one_row(df, axs, False, False)


#### Second
df = pd.read_csv("muraro_value.txt", sep='\t')

ax0 = plt.subplot2grid((5,5), (1,0))
ax1 = plt.subplot2grid((5,5), (1,1))
ax2 = plt.subplot2grid((5,5), (1,2))
ax3 = plt.subplot2grid((5,5), (1,3))
ax4 = plt.subplot2grid((5,5), (1,4))
axs = [ax0, ax1, ax2, ax3, ax4]

one_row(df, axs, False, False)


#### Third
df = pd.read_csv("seg_value.txt", sep='\t')

ax0 = plt.subplot2grid((5,5), (2,0))
ax1 = plt.subplot2grid((5,5), (2,1))
ax2 = plt.subplot2grid((5,5), (2,2))
ax3 = plt.subplot2grid((5,5), (2,3))
ax4 = plt.subplot2grid((5,5), (2,4))
axs = [ax0, ax1, ax2, ax3, ax4]

one_row(df, axs, False, False)


#### Fourth
df = pd.read_csv("wang_value.txt", sep='\t')

ax0 = plt.subplot2grid((5,5), (3,0))
ax1 = plt.subplot2grid((5,5), (3,1))
ax2 = plt.subplot2grid((5,5), (3,2))
ax3 = plt.subplot2grid((5,5), (3,3))
ax4 = plt.subplot2grid((5,5), (3,4))
axs = [ax0, ax1, ax2, ax3, ax4]

one_row(df, axs, False, False)


#### Fifth
df = pd.read_csv("xin_value.txt", sep='\t')

ax0 = plt.subplot2grid((5,5), (4,0))
ax1 = plt.subplot2grid((5,5), (4,1))
ax2 = plt.subplot2grid((5,5), (4,2))
ax3 = plt.subplot2grid((5,5), (4,3))
ax4 = plt.subplot2grid((5,5), (4,4))
axs = [ax0, ax1, ax2, ax3, ax4]

one_row(df, axs, False, True)


# override
override_w_text('Baron', plt.subplot2grid((5,5), (0,0)), True, False)

override_w_text('Muraro', plt.subplot2grid((5,5), (1,1)))
override_w_text('Seg', plt.subplot2grid((5,5), (2,2)))
override_w_text('Wang', plt.subplot2grid((5,5), (3,3)))
override_w_text('Xin', plt.subplot2grid((5,5), (4,4)), False, True)


plt.savefig("tf_naive_"+acc_ari+".png")
