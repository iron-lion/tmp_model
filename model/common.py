import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
#import scanpy
import csv
import glob
import random
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


#string_gene_list_pwd = "./data/genesort_string_hit.txt"


def sample_test_split(geo, num_of_class_test, num_of_example, num_of_testing, string_set, sorting, label_dic=False, pp=False):
    class_folders = geo.keys()
    class_folders = random.sample(class_folders, num_of_class_test)
    if label_dic:
        labels_to_text = label_dic
        labels_converter = {value:key for key, value in label_dic.items()}
    else:
        labels_converter = np.array(range(len(class_folders)))
        labels_converter = dict(zip(class_folders, labels_converter))
        labels_to_text = {value:key for key, value in labels_converter.items()}

    example_set = pd.DataFrame()
    test_set = pd.DataFrame()
    example_label = []
    test_label = []

    # balance sampler
    for subtype in labels_converter.keys():
        this_exp = geo[subtype]
        if (len(this_exp.index.intersection(string_set)) == 0):
            this_exp = this_exp.transpose()
            assert(len(this_exp.index.intersection(string_set)) != 0), "exp array has not symbol"

        this_exp = this_exp[~this_exp.index.duplicated(keep='first')]
        
        total_colno = (this_exp.shape)[1]
        col_nu = list(range(total_colno) )
        random.shuffle(col_nu)
        assert(len(col_nu) >= num_of_example+num_of_testing), [total_colno, num_of_example+num_of_testing, subtype]
        example_ids = col_nu[0 : num_of_example]
        ex = this_exp.iloc[:,example_ids]
        test_ids = col_nu[num_of_example : num_of_example + num_of_testing]
        te = this_exp.iloc[:,test_ids]
        #ex = np.log(ex+1.0)
        #ex = np.clip(ex, 1, np.max(ex)[1])
        #ex = ex.transpose()
        #te = np.log(te+1.0)
        #te = np.clip(te, 1, np.max(te)[1])
        #te = te.transpose()
        example_set = pd.concat([example_set,ex],axis=1)
        test_set = pd.concat([test_set, te],axis=1)
        example_label += [labels_converter[subtype]] * num_of_example
        test_label += [labels_converter[subtype]] * num_of_testing
    
    if string_set is not None:
        example_set = example_set.transpose()
        example_set = example_set.filter(items=string_set)
        example_set = example_set.transpose()

        test_set = test_set.transpose()
        test_set = test_set.filter(items=string_set)
        test_set = test_set.transpose()
    
    out_ex = pd.DataFrame(index=string_set)
    out_ex = pd.concat([out_ex, example_set],axis=1)
    out_ex = out_ex.replace(np.nan,0)

    test_set = test_set.transpose()
    test_set['label'] = test_label
    test_set = test_set.sample(frac=1)
    test_label = test_set['label']
    test_set = test_set.drop(columns='label')
    test_set = test_set.transpose()

    out_te = pd.DataFrame(index=string_set)
    out_te = pd.concat([out_te,test_set], axis=1)
    out_te = out_te.replace(np.nan,0)
    
    if sorting == True:
        out_ex.sort_index(inplace=True)
        out_te.sort_index(inplace=True)
    
    if pp == True:
        print(out_ex.index)
        print(out_te.index)

    assert(np.all(out_ex.index == out_te.index))
    
    if num_of_example != 0:
        out_ex = min_max_scaler.fit_transform(out_ex)
    if num_of_testing != 0:
        out_te = min_max_scaler.fit_transform(out_te)
    
    return out_ex, example_label, out_te, test_label, labels_to_text



def string_gene_symbol_list_load(string_gene_list_pwd, entrez):
    string_set = []
    dic = {}
    with open(string_gene_list_pwd) as open_fd:
        data = csv.reader(open_fd, delimiter=" ")
        
        for rows in data:
            if(rows[2] == "HIT"):
                dic[rows[1]] = rows[0]

    for i in entrez:
        string_set.append(dic[i])
    return string_set


def sorted_string_gene_list_load(string_gene_list_pwd, gene_filter=None):
    string_set = []
    with open(string_gene_list_pwd) as open_fd:
        data = csv.reader(open_fd, delimiter=" ")
        
        for rows in data:
            if(rows[2] == "HIT"):
                if (gene_filter is None) or (rows[1] in gene_filter):
                    string_set.append(rows[0])
                    gene_filter.remove(rows[1])
    return string_set


def string_gene_set_load(string_gene_list_pwd, gene_filter=None):
    string_set = set()
    with open(string_gene_list_pwd) as open_fd:
        data = csv.reader(open_fd, delimiter=" ")
        
        for rows in data:
            if(rows[2] == "HIT"):
                if (gene_filter is None) or (rows[1] in gene_filter):
                    string_set.add(rows[0])

    return string_set


def string_symbol_set_load(string_gene_list_pwd, gene_filter=None):
    string_set = set()
    with open(string_gene_list_pwd) as open_fd:
        data = csv.reader(open_fd, delimiter=" ")
        
        for rows in data:
            if(rows[2] == "HIT"):
                if (gene_filter is None) or (rows[1] in gene_filter):
                    string_set.add(rows[1])

    return string_set


""" copied stat code"""
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data).astype(np.float)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    
    return m,h


###
#  SINGLE CELL PANCREAS DATA  #
                            ###

def geo_data_loader(root_dir, pref=None, string_set=None, small_set=False):
    # merged
    # single_cell pancreas data scan/read
    pd_list = dict()

    file_list = glob.glob(root_dir + "*.csv.pd")
    if small_set == True:
        file_list = file_list + glob.glob(root_dir + "*.csv.pd.bak")
    #print(file_list)
    for type_data in file_list:
        this_pd = pd.read_pickle(type_data)
        class_name = type_data[len(root_dir) : -len(".csv.pd")]
        if pref is not None:
            class_name = class_name + "_" + str(pref)
        
        if (string_set is not None):

            if (len(this_pd.index.intersection(string_set)) == 0):
                this_pd = this_pd.transpose()
                assert(len(this_pd.index.intersection(string_set)) != 0), "exp array has not symbol"
            
            #print('ori', this_pd.shape, this_pd.index)
            this_pd = this_pd[~this_pd.index.duplicated(keep='first')]
            this_pd = this_pd.transpose()
            this_pd = this_pd.filter(items=string_set)
            this_pd = this_pd.transpose()

            out_pd = pd.DataFrame(index=string_set)
            out_pd = pd.concat([out_pd, this_pd],axis=1)
            out_pd = out_pd.replace(np.nan,0)

            out_pd = out_pd.divide(out_pd.sum(0), axis = 1).mul(20000)          
            out_pd = out_pd.replace(np.nan,0)

            #out_pd = np.log2(out_pd+1.0)
            pd_list[class_name] = out_pd
        else:
            assert(False)
        #print(this_pd.shape)
    return pd_list


def split_geo(geo, number_of_sample, number_of_class, tr, seed, sample_limit=0):
    classes = geo.keys()
    random.seed(seed)

    classes = list(classes)
    random.shuffle(classes)
    random.seed()
    cnt = 0
    pd_list = dict()
    for c in classes:
        if (cnt >= number_of_class):
            break
        this_exp = geo[c]
        if(tr == True):
            this_exp = this_exp.transpose()
        total_colno = (this_exp.shape)[1]
        if (sample_limit > 0) and (total_colno < sample_limit):
            continue
        else :
            print(c)
        cnt += 1
        
        col_nu = list(range(total_colno) )
        random.shuffle(col_nu)
        ns = min(len(col_nu), number_of_sample)
        example_ids = col_nu[0 : ns]
        ex = this_exp.iloc[:,example_ids]
        pd_list[c] = ex
        
        this_exp.drop(this_exp.columns[example_ids], axis=1, inplace=True)
    return pd_list
