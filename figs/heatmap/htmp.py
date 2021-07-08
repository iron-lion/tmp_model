import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv("gtex_15_baron_105_tmp.csv", sep=',', index_col = 0)
#df.fillna(0.0, inplace=True)
df.sort_index(axis=0, inplace=True)
df.sort_index(axis=1, inplace=True)
print(df)
plt.figure(figsize=(600,600), dpi=600)
fig, ax = plt.subplots()
#df = np.log2(df+1)
df = df.div(df.sum(axis=1), axis=0) 
df_np = df.to_numpy()
df_np2 = df_np
print(df_np2)
im = ax.imshow(df_np2, cmap='pink')

# We want to show all ticks...
ax.set_xticks(np.arange(df.shape[1]))
ax.set_yticks(np.arange(df.shape[0]))
# ... and label them with the respective list entries
ax.set_xticklabels(df.columns, fontsize=6)
ax.set_yticklabels(df.index, fontsize=6)

plt.setp(ax.get_xticklabels(), rotation=50, ha="right",
         rotation_mode="anchor")

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        t = df_np[i,j]
        t = round(t, 2)
        if t == 0.0:
            t = 'NaN'
            continue
#        text = ax.text(j, i, t, ha="center", va="center", color="w")

#plt.ylabel('number of data per class')
#plt.xlabel('class number used in training')
fig.tight_layout()
plt.savefig("heatmap.png")

