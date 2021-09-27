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

BARON_DIR = str("./data/scPan/baron/CLASS_WISE/")
MURARO_DIR = str("./data/scPan/muraro/CLASS_WISE/")
XIN_DIR = str("./data/scPan/xin/CLASS_WISE/")
SEG_DIR = str("./data/scPan/segerstolpe/CLASS_WISE/")
WANG_DIR =  str("./data/scPan/wang/CLASS_WISE/")

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
    
    # training code block, fine-tune
    geo = common.geo_data_loader(TRAIN_DIR, 0, string_set)

    if params.split_sample > 0:
        geo = common.split_geo(geo, params.split_sample, len(geo.keys()), False, params.manual_seed)
    
    tmp.train(geo, string_set, False)

    PAN_LIST.remove(TRAIN_DIR)
    # training code block end
    
    i = 0
    for target_dir in PAN_LIST:
        i+=1
        print(target_dir, i)
        testgeo = common.geo_data_loader(target_dir, i, string_set, False)
        tmp.test(testgeo, string_set, False)
    

if __name__ == '__main__':
    print(TRAIN_DIR)
    main()

