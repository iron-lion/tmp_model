import glob
import os
import random
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch
from model import common as common
from model.tmp_model import TMP
from args_parser import get_parser

BARON_DIR = str("./data/scPan/baron/")
MURARO_DIR = str("./data/scPan/muraro/")
XIN_DIR = str("./data/scPan/xin/")
SEG_DIR = str("./data/scPan/segerstolpe/")
WANG_DIR =  str("./data/scPan/wang/")

PAN_LIST = [BARON_DIR, MURARO_DIR, XIN_DIR, SEG_DIR, WANG_DIR]
TRAIN_DIR = BARON_DIR


def main():
    params = get_parser().parse_args()
    print(params)
    params.device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'

    string_set = common.string_symbol_set_load(None)
    string_set = list(string_set)
    string_set.sort()
    tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)

    geo = common.geo_data_loader(TRAIN_DIR, 0, string_set)

    if CAT_LIMIT > 0:
        geo = common.split_geo(geo, params.split_sample, len(geo.keys()), True, params.manual_seed)
    
    PAN_LIST.remove(TRAIN_DIR)
    tmp.train(geo, string_set, True)

    test_all = {}
    i = 0
    for target_dir in PAN_LIST:
        i+=1
        print(target_dir, i)
        testgeo = common.geo_data_loader(target_dir, i, string_set, False)
        test_all.update(testgeo)
    print(test_all.keys())
    tmp.test_with_histogram(test_all, string_set, True, 'gtex_15_baron_105_tmp') #output historgram file name
    

if __name__ == '__main__':
    main()
    print(TRAIN_DIR)
