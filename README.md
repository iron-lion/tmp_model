# data_loader

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
gtex_training.py --cuda -nepoch=300000 -lr=0.0005 -lrS=100000 -lrG=0.5 -fe=<FILENAME> -rn=<FILE_NAME>
### GTEx baseline model params
    5way-task
    Batch_size  = 10
    Episode 	= 300000
    LR      	= 0.0005
    Step_size, Gamma = 100000, 0.5
    non-negative clamp : feature encoder (True) / relation network (False)


## TCGA
tcga_test.py --cuda  -fe=<FILENAME> -rn=<FILE_NAME>
### TCGA model params


## Single-cell pancreas data run
sc_testing.py --cuda   -fe=<FILENAME> -rn=<FILE_NAME>
### Single-cell pancreas data params
    5way-task
    Test runs = 500
    Test batch_size = 10
    LR = 0.0001
    Training subset = 15
    Step_size, Gamma = 100000, 0.5

