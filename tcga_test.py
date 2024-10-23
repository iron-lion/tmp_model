import random
import os
import pandas as pd 
import numpy as np
import torch
from model import common as common
from model.tmp_model import TMP
from model import tcga_loader as tcga_loader
from args_parser import get_parser


string_gene_list_pwd = "./data/genesort_string_hit.txt"


def main():
    params = get_parser().parse_args()
    print(params)
    params.device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    
    string_set = common.string_symbol_set_load(string_gene_list_pwd, None)
    string_set = common.sorted_string_gene_list_load(string_gene_list_pwd, string_set)
 
    tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)

    tcga = tcga_loader.TCGALoader('/home/parky/data/TCGA/', string_set)
    tcga.read_all()
    
    training_geo = common.split_geo(tcga.geo, params.split_sample, params.split_class, False, 3, 200)

    tmp.train(training_geo, string_set, False)
    tmp.test(tcga.geo, string_set, False)

if __name__ == '__main__':
    main()
