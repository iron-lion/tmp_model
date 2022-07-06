# TMP_model
Transferable Molecular Pattern recognition model.

<img src="https://github.com/iron-lion/tmp_model/blob/master/figs/poster_brief%20(1).png " width="400"> _Created with BioRender.com_

[bioRxiv: Transfer Learning Compensates Limited Data, Batch-Effects, And Technical Heterogeneity In Single-Cell Sequencing](https://doi.org/10.1101/2021.07.23.453486)


## Reproducibility
* Models pretrained with GTEx and TCGA are available in this [link](https://zenodo.org/record/5529755#.YVGe4bozawF).
* Datasets used in this study are described in the [data folder](https://github.com/iron-lion/tmp_model/tree/master/data).
* All processed datasheets and figure generating codes are in the [figures folder](https://github.com/iron-lion/tmp_model/tree/master/figs)

Details
1. [GTEx result](https://github.com/iron-lion/tmp_model#gtex)
2. [TCGA result](https://github.com/iron-lion/tmp_model#TCGA)
3. [scPancreas result](https://github.com/iron-lion/tmp_model#Single-cell_pancreas)



## Environment
    conda version : 4.9.2
    conda-build version : 3.20.5
    python version : 3.8.5.final.0
    virtual packages : __cuda=11.2=0
                       __glibc=2.31=0
                       __unix=0=0
                       __archspec=1=x86_64
    scikit-learn : 0.23.2     
    scipy : 1.6.2      
    torch : 1.9.0a0+gita80b215 



## GTEx

### GTEx baseline model params
    5way-task
    Batch_size  = 10
    Episode 	= 300000
    LR      	= 0.0005
    Step_size, Gamma = 100000, 0.5
    non-negative clamp : feature encoder (True) / relation network (False)

```python
# baseline model is trained with this parameters.
# len(string_set) = 18,000
tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
```
   
```
python gtex_training.py --cuda -nepoch=300000 -lr=0.0005 -lrS=100000 -lrG=0.5 -fe=<FILENAME> -rn=<FILE_NAME>
```

## TCGA

### When pretrain model with TCGA dataset.

#### TCGA model params
    * Pre-trained model is produced with same parameters with GTEx model.
    5way-task
    Batch_size  = 10
    Episode 	= 300000
    LR      	= 0.0005
    Step_size, Gamma = 100000, 0.5
    non-negative clamp : feature encoder (True) / relation network (False)

```python
 # in the tcga_test.py file
 20  tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
 21                                                                                 
 22  tcga = tcga_loader.TCGALoader('/home/parky/data/TCGA/', string_set)         
 23  tcga.read_all()                                                             
 24                                                                                 
 25  training_geo = common.split_geo(tcga.geo, params.t1, params.t2, False, 3, 0) # Last argument is a minimum size of data 
 26                                                                                 
 27  tmp.train(training_geo, string_set, False)
```
In case pretraining, used all cancer types in TCGA dataset.
```
python tcga_test.py --cuda -nepoch=300000 -lr=0.0005 -lrS=100000 -lrG=0.5 -fe=<FILENAME> -rn=<FILE_NAME>
```

### When test GTEx pretrained model with TCGA dataset.

```python
 # in the tcga_test.py file
 20  tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
 21                                                                                 
 22  tcga = tcga_loader.TCGALoader('/home/parky/data/TCGA/', string_set)         
 23  tcga.read_all()                                                             
 24                                                                                 
 25  #training_geo = common.split_geo(tcga.geo, params.t1, params.t2, False, 3, 200)
 26                                                                                 
 27  #tmp.train(training_geo, string_set, False)                                 
 28  tmp.test(tcga.geo, string_set, False) # run with test
```
In case test, used all cancer types in TCGA dataset.
```
python tcga_test.py --cuda -fe=<GTEx_FILENAME> -rn=<GTEx_FILE_NAME> -nepoch_test=5000
```

### Fine-tuning training with TCGA and test
```python
 # in the tcga_test.py file
 20  tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
 21                                                                                 
 22  tcga = tcga_loader.TCGALoader('/home/parky/data/TCGA/', string_set)         
 23  tcga.read_all()                                                             
 24                                                                                 
 25  training_geo = common.split_geo(tcga.geo, params.t1, params.t2, False, 3, 200)
 26                                                                                 
 27  tmp.train(training_geo, string_set, False)                                 
 28  tmp.test(tcga.geo, string_set, False) # run with test
```
Case) ''S'' samples for all 33 class in TCGA dataset.
```
python tcga_test.py --cuda -fe=<GTEx_FILENAME> -rn=<GTEx_FILE_NAME> -nepoch=50000 -nepoch_test=5000 -lr=0.0005 -lrS=1500 -lrG=0.5 -logb=2500 --early_stop1 --split_sample=<S> --split_class=33
```
Case) 15 samples for all ''N'' class in TCGA dataset.
```
python tcga_test.py --cuda -fe=<GTEx_FILENAME> -rn=<GTEx_FILE_NAME> -nepoch=50000 -nepoch_test=5000 -lr=0.0005 -lrS=1500 -lrG=0.5 -logb=2500 --early_stop1 --split_sample=15 --split_class=<N>
```

## Single-cell_pancreas
single-cell datasets are hard-coded in the sc_training.py file.

```python
 12 BARON_DIR = str("./data/scPan/baron/CLASS_WISE/")                               
 13 MURARO_DIR = str("./data/scPan/muraro/CLASS_WISE/")                             
 14 XIN_DIR = str("./data/scPan/xin/CLASS_WISE/")                                   
 15 SEG_DIR = str("./data/scPan/segerstolpe/CLASS_WISE/")                           
 16 WANG_DIR =  str("./data/scPan/wang/CLASS_WISE/")                                
 17                                                                                 
 18 PAN_LIST = [BARON_DIR, MURARO_DIR, XIN_DIR, SEG_DIR, WANG_DIR]                  
 19 TRAIN_DIR = BARON_DIR
 ...
 42     for target_dir in PAN_LIST:                                                 
 43         ... # TEST CODE BLOCK
```

### When test GTEx pretrained model with TCGA dataset.
```python
30  tmp = TMP(params, len(string_set), e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100)
31  """                                                                         
32  geo = common.geo_data_loader(TRAIN_DIR, 0, string_set)                      
33                                                                                 
34  if params.split_sample > 0:                                                 
35      geo = common.split_geo(geo, params.split_sample, len(geo.keys()), True, params.manual_seed)
36                                                                                  
37  tmp.train(geo, string_set, False)                                           
38                                                                                 
39  PAN_LIST.remove(TRAIN_DIR)                                                   
40  """                                                                         
41  i = 0                                                                       
42  for target_dir in PAN_LIST:                                                  
43      i+=1                                                                    
44      print(target_dir, i)                                                     
45      testgeo = common.geo_data_loader(target_dir, i, string_set, False)      
46      tmp.test(testgeo, string_set, False)  
```
Make it skip training codeblock.
```
python sc_training.py --split_sample=0 -nexample=5 -nbatch=5 -nepoch_test=5000 -fe=<GTEx_FILE_NAME> -rn=<GTEx_FilE_NAME>
```

### Fine-tuning training with subset of single-cell pancreas datasets and test

In the article, ''S'' is set to 15.

```
python sc_training.py --split_sample=<S> -nexample=5 -nbatch=5 -nepoch=50000 -logb=200 -nepoch_test=2500 -lr=0.0001 -lrS=10000 -fe=<FILE_NAME> -rn=<FILE_NAME>
```


### Draw heatmap of few-shot classification result with single-cell Pancreas dataset.
For N-way K-shot test, set ''N'' and ''K'' to (5, 5), (10, 5), or (20, 5).
```
python sc_histogram.py -nexample=<K> -nbatch=5 -nclass=<N> -nepoch_test=5000 -fe=<FILE_NAME> -rn=<FILE_NAME>
```

