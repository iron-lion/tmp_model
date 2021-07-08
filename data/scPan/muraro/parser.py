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

file_list = glob.glob("./RAW/data.csv")
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

new_index = []
tmp_index = total_data.index
for ll in tmp_index:
    new_index.append(ll.split("__")[0])
total_data.index = new_index

print(total_data)

labels = pd.read_csv('./RAW/cell_type_annotation_Cels2016.csv', sep='\t', index_col=0)
labels.columns = ['labels']
label_colname = labels.columns[0]
print(label_colname)
labels.index = [l.replace(".","-") for l in labels.index]
labels = labels.transpose().filter(items=total_data.columns)
print(labels)

total_data = total_data.append(labels)
total_data=total_data.transpose()
print(total_data)

total_data[label_colname] = total_data[label_colname].replace(np.nan, 'nolabel')


label_set = set(total_data[label_colname])
print(label_set)
for l in label_set:
    type_data = total_data.loc[total_data[label_colname] == l,:]

    tmp = type_data
    tmp.pop(label_colname)
    tmp = tmp + 1.0
    tmp = tmp.applymap(np.log2)
    tmp = tmp.transpose()
    tmp.sort_index(inplace=True)
    tmp.to_pickle(CLA_DIR+str(l)+'.csv.pd')

    print(l, type_data.shape)

sys.stdout = ori_stdout
