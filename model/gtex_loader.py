import glob
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn

min_max_scaler = preprocessing.MinMaxScaler()

class GTExTaskMem(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, whole_dic,data_root_dir, num_classes, train_num, test_num):

        self.character_folders = whole_dic.keys()
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        
        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip([i[len(data_root_dir):] for i in class_folders], labels))

        self.train_roots = []
        self.test_roots = []
        self.train_labels = []
        self.test_labels = []
 
        for types in class_folders:
            this_tmp = whole_dic[types]
            len_col = this_tmp.shape[1]
            col_nu = list(range(len_col) )
            random.shuffle(col_nu)
            
            self.train_roots.append(col_nu[:train_num])
            self.test_roots.append( col_nu[train_num:test_num + train_num])
            self.train_labels.append([labels[types]] * train_num)
            self.test_labels.append([labels[types]] * test_num)
 
        self.label_set = self.class_folders
        self.label_list = list(self.label_set)
            

def sample_test_split(geo, num_of_class_test, num_of_example, num_of_testing, string_set, tr):
    class_folders = geo.keys()
    class_folders = random.sample(class_folders, num_of_class_test)
    labels_converter = np.array(range(len(class_folders)))
    labels_converter = dict(zip(class_folders, labels_converter))
    labels_to_text = {value:key for key, value in labels_converter.items()}

    example_set = pd.DataFrame()
    test_set = pd.DataFrame()
    example_label = []
    test_label = []

    # balance sampler
    for ith in range(len(class_folders)):
        subtype = class_folders[ith]
        this_exp = geo[subtype]
        if(tr == True):
            this_exp = this_exp.transpose()
        total_colno = (this_exp.shape)[1]
        col_nu = list(range(total_colno) )
        random.shuffle(col_nu)
        assert(len(col_nu) > num_of_example+num_of_testing), subtype
        example_ids = col_nu[0 : num_of_example]
        ex = this_exp.iloc[:,example_ids]
        test_ids = col_nu[num_of_example : num_of_example + num_of_testing]
        te = this_exp.iloc[:,test_ids]

        ex.sort_index(inplace=True)
        te.sort_index(inplace=True)
        example_set = pd.concat([example_set,ex],axis=1)
        test_set = pd.concat([test_set, te],axis=1)
        example_label += [labels_converter[subtype]] * num_of_example
        test_label += [labels_converter[subtype]] * num_of_testing
    
    if string_set is not None:
        example_set = example_set[~example_set.index.duplicated(keep='first')]
        example_set = example_set.transpose()
        example_set = example_set.filter(items=string_set)
        example_set = example_set.transpose()
        test_set = test_set[~test_set.index.duplicated(keep='first')]
        test_set = test_set.transpose()
        test_set = test_set.filter(items=string_set)
        test_set = test_set.transpose()
    out_ex = pd.DataFrame(index=string_set)
    out_ex = pd.concat([out_ex, example_set],axis=1)
    out_ex = out_ex.replace(np.nan,0)
    #out_ex.sort_index(inplace=True)

    test_set = test_set.transpose()
    test_set['label'] = test_label
    test_set = test_set.sample(frac=1)
    test_label = test_set['label']
    test_set = test_set.drop(columns='label')
    test_set = test_set.transpose()
    out_te = pd.DataFrame(index=string_set)
    out_te = pd.concat([out_te,test_set], axis=1)
    #out_te.sort_index(inplace=True)
    out_te = out_te.replace(np.nan,0)
    
    out_ex = min_max_scaler.fit_transform(out_ex)
    out_te = min_max_scaler.fit_transform(out_te)
    return out_ex, example_label, out_te, test_label, labels_to_text



##### relation learning implementation

class GTExTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, data_root_dir, num_classes, train_num, test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        
        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip([i[len(data_root_dir):] for i in class_folders], labels))
        self.train_roots = []
        self.test_roots = []
        self.train_labels = []
        self.test_labels = []
               
        label_set = set()
        for c in class_folders:
            label_string = c[len(data_root_dir):]
            label_set.add(label_string)
            label_file_list = glob.glob(c + "/*")
            label_file_list = random.sample(label_file_list,len(label_file_list))
            
            self.train_roots += label_file_list[:train_num]
            self.test_roots += label_file_list[train_num:test_num + train_num]
            self.train_labels += [labels[label_string]] * train_num
            self.test_labels += [labels[label_string]] * test_num
        self.label_set = label_set      
        #label size dic done
        self.label_list = list(self.label_set)
             

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])
 
 
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst,shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i+j*self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1


class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1



 
class ClassBalancedSamplerMem(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst, num_set):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.num_set = num_set #[(0,10), (1,10),(2,10)]
        print(self.num_set)

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        picked = random.sample(range(len(self.num_set)), self.num_cl)
        batches = [[ int(i * 10000 + j) for j in random.sample(range(self.num_set[i]), self.num_inst)] for i in picked]
        batches = [item for sublist in batches for item in sublist]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return 1


# need to split file list 
class GTExGepDatasetMem(torch.utils.data.Dataset):

    def __init__(self, geo_dic, gene_set):
        super(GTExGepDatasetMem, self).__init__()

        self.geo_dic = geo_dic
        self.keys = []
        self.num_per_classes = []
        for i in self.geo_dic:
            self.keys.append(i)
            self.num_per_classes.append((geo_dic[i].shape)[1])
        self.gene_set = gene_set

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        print('>')
        print(idx)
        print('<')
        i = idx//10000
        j = idx%10000
        ki = self.keys[i]
        gep = self.geo_dic[ki].iloc[:,j]
        return gep, ki


def get_gtex_loader_mem(tr_data, num_class, num_per_class=5, gene_set=None):
    # NOTE: batch size here is # instances PER CLASS
    #normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = GTExGepDatasetMem(tr_data, gene_set)
    sampler = ClassBalancedSamplerMem(num_class. num_per_class, dataset.num_per_classes)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    
    return loader
 


# need to split file list 
class GTExGepDataset(torch.utils.data.Dataset):

    def __init__(self, task, split, gene_set, headers):
        if (split == "train"):
            self.file_list = task.train_roots
            self.label_list = task.train_labels
        else:
            self.file_list = task.test_roots
            self.label_list = task.test_labels
        self.gene_set = gene_set
        self.headers = headers
        
    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        data = pd.read_csv(file_name, sep='\t' ,index_col=0,  header=None) 
        data = np.log(data+1.0)
        #data = np.clip(data, 1, np.max(data)[1])
        data = data.transpose()
        data.columns = [i.split(".")[0] for i in data.columns]
        data = data.filter(items=self.gene_set)
        data = data.transpose()

        out_ex = pd.DataFrame(index=self.gene_set)
        out_ex = pd.concat([out_ex, data],axis=1)
        out_ex = out_ex.replace(np.nan,0)
        out_ex = np.array(out_ex).reshape(-1,1)
 
        #data.sort_index(inplace = True)
        #data = data.astype('float')

        
        data_label = self.label_list[idx]
        data_label = np.array(data_label)
        data_label = data_label.astype('int')

        return out_ex, data_label


def get_gtex_loader(task, num_per_class=1, split='train',shuffle=True, gene_set=None, order=None):
    # NOTE: batch size here is # instances PER CLASS
    #normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
        dataset = GTExGepDataset(task, split, gene_set, order)

    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num,shuffle=shuffle)    
        dataset = GTExGepDataset(task, split, gene_set, order)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    
    return loader
 

def GTEx_divide_train_test(root_dir, num_class, seed_num):      
    
    metatrain_character_folders = glob.glob(root_dir + "training_gtex/*")
    metaval_character_folders = glob.glob(root_dir + "test_gtex/*")
    print ("We are training :", metatrain_character_folders)
    print ("We are testing :", metaval_character_folders)
    return  metatrain_character_folders, metaval_character_folders


def GTEx_divide_train_test_all(root_dir, num_class, seed_num):      
    tissue_folder = glob.glob(root_dir + "*")
    
    random.seed(seed_num)
    random.shuffle(tissue_folder)
    
    metatrain_character_folders = tissue_folder[num_class:]
    metaval_character_folders = tissue_folder[:num_class] #first num_class is for test
    print ("We are training :", metatrain_character_folders)
    print ("We are testing :", metaval_character_folders)
    return  metatrain_character_folders, metaval_character_folders
        
def GTEx_divide_train_test_all_in_memory(root_dir, num_class, seed_num):

    tissue_folder = glob.glob(root_dir + "sub_*")
        
    #random.seed(seed_num)
    random.shuffle(tissue_folder)
    
    metatrain_character_folders = tissue_folder[num_class:]
    metaval_character_folders = tissue_folder[:num_class] #first num_class is for test
    print ("We are training :", metatrain_character_folders)
    print ("We are testing :", metaval_character_folders)

    meta_train = dict()
    meta_test = dict()

    for type_data in metatrain_character_folders:
        this_pd = pd.read_csv(type_data,sep="\t",index_col=0, header=None)
        this_pd.index = [i.split(".")[0] for i in this_pd.index]
        this_pd.columns = [type_data+"_"+str(i) for i in this_pd.columns ]
        class_name = type_data[len(root_dir) : ]
        this_pd = np.log(this_pd.add(1.0))
        meta_train[class_name] = this_pd

    for type_data in metaval_character_folders:
        this_pd = pd.read_csv(type_data,sep="\t",index_col=0, header=None)
        this_pd.index = [i.split(".")[0] for i in this_pd.index]
        this_pd.columns = [type_data+"_"+str(i) for i in this_pd.columns ]
        class_name = type_data[len(root_dir) : ]
        this_pd = np.log(this_pd.add(1.0))
        meta_train[class_name] = this_pd#.transpose()

    return  meta_train, meta_test
     
