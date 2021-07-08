import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("small_pca.txt", sep='\t')


plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()

#ax.plot(df['per_sample'], df['ARI'], yerr=df['ARI_95']) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(df['per_sample'], df['ARI'], color='blue') 

ax.plot(29, 0.4578722005, '--bo',color='blue') # full data

ax.fill_between(df['per_sample'], df['ARI']-df['ARI_95'], df['ARI']+df['ARI_95'], facecolor='green', alpha=0.3)

ax.set_xticklabels([0,5,10,15,20,25,'All'])
plt.xlabel('Number of annotated cells per class')
plt.ylabel('ARI')

plt.axis([0,31,0,0.51])
plt.savefig("ari.png")
