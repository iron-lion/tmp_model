import glob
import os
import random
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch
from model import common as common
from model import gtex_loader as gtex_loader
from model.tmp_model import TMP
from args_parser import get_parser


string_gene_list_pwd = './data/genesort_string_hit.txt'

def main():
    params = get_parser().parse_args()
    params.device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    print(params)

    string_set = common.string_symbol_set_load(string_gene_list_pwd,, None)
    string_set = common.sorted_string_gene_list_load(string_gene_list_pwd, string_set)

    tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
    
    train_data_dic, test_data_dic = gtex_loader.GTEx_divide_train_test_all_in_memory("./data/GTEx/", 0, 0)
    
    tmp.train(train_data_dic, string_set, True)

if __name__ == '__main__':
    main()
