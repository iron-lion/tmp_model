import pandas as pd
import numpy as np
import glob
import os
import sys

ori_stdout = sys.stdout
f = open('parser_log', 'w')
sys.stdout=f


CLA_DIR = "./CLASS_WISE/"
if not os.path.exists(CLA_DIR):
    os.mkdir(CLA_DIR)

LABEL = 'Characteristics[cell_type]'

file_list = glob.glob("./RAW/cut_exp.csv")
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

labels = pd.read_csv('./RAW/E-MTAB-5061.sdrf.txt', sep='\t', index_col=0)
labels = labels[LABEL]
total_data = total_data.iloc[:,1:]
total_data = total_data.append(labels.transpose())
print(total_data)
total_data=total_data.transpose()

print(total_data[LABEL])
label_set = set(total_data[LABEL])
print(label_set)
for l in label_set:
    type_data = total_data.loc[total_data[LABEL] == l,:]

    tmp = type_data
    tmp.pop(LABEL)
    tmp = tmp + 1.0
    tmp = tmp.applymap(np.log2)
    tmp = tmp.transpose()
    tmp.sort_index(inplace=True)
    print(tmp.index)
    tmp.to_pickle(CLA_DIR+str(l)+'.csv.pd')

    print(l, type_data.shape)

sys.stdout = ori_stdout
