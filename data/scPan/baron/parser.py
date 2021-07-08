import pandas as pd
import numpy as np
import os
import glob
import sys

ori_stdout = sys.stdout
f = open('parser_log', 'w')
sys.stdout=f


CLA_DIR = "./CLASS_WISE/"
if not os.path.exists(CLA_DIR):
    os.mkdir(CLA_DIR)

file_list = glob.glob("./RAW/*human*counts.csv")
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep=',', index_col=0)
    total_data = total_data.append(data)
#    print(total_data.shape)


labels = total_data['assigned_cluster']
label_set = set(labels)
print(label_set)

for l in label_set:
    type_data = total_data.loc[labels==l,:]

    tmp = type_data.iloc[:,2:]
    tmp = tmp.transpose()
    tmp = tmp + 1.0
    tmp = tmp.applymap(np.log2)
    tmp.sort_index(inplace=True)
    tmp.to_pickle(CLA_DIR + l + '.csv.pd')
    #print(tmp.index)
    print(l, type_data.shape)

sys.stdout = ori_stdout
