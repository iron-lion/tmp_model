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

gene_anno = pd.read_csv('./RAW/human_gene_annotation.csv', sep=",", index_col=0)

LABEL = 'cell.type'

file_list = glob.glob("./RAW/data.txt")
print(file_list)

total_data = pd.DataFrame()
for ff in file_list:
    data = pd.read_csv(ff, sep='\t', index_col=0)
    total_data = total_data.append(data)
    print(total_data.shape)

total_data = pd.concat([gene_anno, total_data], axis=1)
total_data.set_index('symbol', inplace=True)
print(total_data)


labels = pd.read_csv('./RAW/human_islet_cell_identity.txt', sep='\t', index_col=0)
labels = labels[LABEL]

#print(total_data)
#total_data = total_data.iloc[:,1:]
#total_data.columns = [k.split(".")[1] for k in total_data.columns]

total_data = total_data.append(labels)
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
    tmp.sort_index(inplace = True)
    tmp.to_pickle(CLA_DIR+str(l)+'.csv.pd')

    print(l, type_data.shape)

sys.stdout = ori_stdout
