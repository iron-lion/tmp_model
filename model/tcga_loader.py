import os
import sys
import csv
import json
import glob
import random
import pandas as pd
import numpy as np


class TCGALoader():
    def __init__(self, root_dir, string_set):
        self.root_dir  = root_dir
        self.string_set = string_set
        self.type_num  = 0
        self.type_list = []
        self.path_dic  = {}
        self.file_dic = []
        self.geo = {}

    def initial_work(self):
        rm_list = glob.glob(self.root_dir + '*.csv.h5')
        type_list = glob.glob(self.root_dir + 'tcga*')
        [type_list.remove(a) for a in rm_list]
        self.type_num = len(type_list)
        for types in type_list:
            self.type_list.append(types[len(self.root_dir): ] )
            m_p, e_f, b_p = self.three_data_checker(self.root_dir + types[len(self.root_dir): ])
            self.path_dic[types[len(self.root_dir): ]] = (m_p, e_f, b_p)
        
        self.file_dic = self.tcga_read_prepare()
        del(self.path_dic)
        
        self.load_all()
        del(self.file_dic)
        print(self.geo)


    def tcga_read_prepare(self, is_normal=False):
        sampled_file_dic = {}
        for kp in self.path_dic.keys(): # N cut
            (m_p, e_f, b_p) = self.path_dic[kp]
            md = self.meta_reader(m_p)
            cd = self.case_reader(b_p)
            file_name_list_got = (self.exp_reader(e_f, md, cd, '', is_normal))
            if len(file_name_list_got) > 10:
                sampled_file_dic[kp] = file_name_list_got
        return sampled_file_dic


    def load_all(self):
        all_class = self.file_dic.keys()

        for cl in all_class:
            class_file_path = self.root_dir + cl + '.csv.h5'
            if os.path.exists(class_file_path):
                #print('loading... HDF5',cl)
                self.geo[cl] = pd.read_hdf(class_file_path, "table")
            else :
                #print('loading... RAW',cl)
                f_l = self.file_dic[cl]
                dfs = pd.DataFrame(index=string_set)
                for f in f_l:
                    ndf = pd.read_csv(self.root_dir + cl + '/exp/' + f[0], sep='\t' ,index_col=0, header=None)
                    ndf.index = [i.split(".")[0] for i in ndf.index]
                    ndf.columns = [f[0].rstrip(".FPKM.txt")]
                    dfs = pd.concat([dfs, ndf], axis=1)
                dfs = dfs.filter(items=string_set, axis=0)
                dfs.to_hdf(class_file_path, "table")
                self.geo[cl] = dfs
            self.file_dic[cl] = []


    def read_all(self):
        h5_list = glob.glob(self.root_dir + '*.csv.h5')
        assert(len(h5_list)==33), 'missing tcga-h5 files, to initial work first'
        for hf in h5_list:
            class_label = hf[len(self.root_dir):-len('.csv.h5')]
            df_table = pd.read_hdf(hf, "table") + 1.0
            self.geo[class_label] = df_table.apply(np.log2)
            #print('loading... HDF5',hf, df_table.shape)
        #print(self.geo.keys())


    def three_data_checker(self, path):
        f_list = glob.glob(path + '/*')
        metadata_path, expression_folder, bio_path = '', '', ''

        for files in f_list:
            if ("metadata" in files):
                metadata_path = files
            elif ("exp" in files):
                expression_folder = files
            elif ("sample.tsv" in files):
                bio_path = files
        if ((len(metadata_path) > 0) and (len(expression_folder) > 0) and (len(bio_path) > 0)):
            return metadata_path, expression_folder, bio_path
        else:
            print("Err: ", path, len(metadata_path), len(expression_folder), len(bio_path))
            return '','',''


    def meta_reader(self, path):    
        file_to_case_dic = {}
        with open(path) as json_file:
            data = json.load(json_file)
            for case in data:
                file_to_case_dic[case['file_name'][:-3]] = case['associated_entities'][0]['entity_submitter_id'][:16]
               
        #print(len(file_to_case_dic))
        return file_to_case_dic


    def case_reader(self, path):
        case_to_meta = {}
        with open(path) as meta_file:
            data = csv.reader(meta_file, delimiter='\t')
            header = next(data)
            type_column = header.index('sample_type')
            id_column   = header.index('sample_submitter_id')
            case_to_meta = {rows[id_column]:rows[type_column] for rows in data}
            
        #print(len(case_to_meta), type_column, id_column)
        return case_to_meta
       

    def exp_reader(self, path, file_to_case_dic, case_to_meta, subtypes, is_normal):
        fd = glob.glob(path + '/*/*.FPKM.txt')
        file_name_list = []
        
        Tumor, Tissue, Blood = 0,0,0

        for each_file in fd:
            k = each_file.split('/')
            each_file_name = k[-2]+'/'+k[-1]
            
            data_type = case_to_meta[file_to_case_dic[each_file_name.split('/')[1]]]
            subtype=''
            try:
                subtype = subtypes[file_to_case_dic[each_file_name.split('/')[1]][:12]]
            except:
                subtype = ''
            
            descrete_type = -1
            if ("Tumor" in data_type):
                Tumor +=1
                descrete_type = 0
            elif ("Tissue" in data_type):
                Tissue +=1
                descrete_type = -1
            elif ("Blood" in data_type):
                Blood +=1
                descrete_type = 22
            else:
                continue
            
            if (is_normal is True) and (descrete_type == -1) \
                or ((is_normal is not True) and (descrete_type == 0)) \
                or ((is_normal is not True) and (descrete_type == 22)):
                try:
                    file_name_list.append((each_file_name, descrete_type))
                except:
                    print('could not found meta information FILE_PATH: ', each_file_name.split('/')[1])

        #print(path, ' - Data shape: Tumor: ',Tumor,' Tissue: ', Tissue, ' Blood: ', Blood)
        return file_name_list


# TCGA exp loader end 


if __name__=='__main__':
    import common as common
    spwd = "./data/genesort_string_hit.txt"
    string_set = common.string_symbol_set_load(spwd)
    string_set = list(string_set)
    string_set.sort()
    string_set = common.sorted_string_gene_list_load(spwd, string_set)
    
    tcga = TCGALoader('./data/TCGA/', string_set)
    tcga.initial_work()
    #tcga.read_all()
