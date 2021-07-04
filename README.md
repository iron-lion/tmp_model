# data_loader

## GTEx
gtex_training.py --cuda -nepoch=300000 -lr=0.0005 -lrS=100000 -lrG=0.5
### GTEx baseline model params
    5way-task
    Batch_size  = 10
    Episode 	= 300000
    LR      	= 0.0005
    Step_size, Gamma = 100000, 0.5
    non-negative clamp : feature encoder (True) / relation network (False)


## TCGA
tcga_test.py --cuda
### TCGA model params


## Single-cell pancreas data run
sc_testing.py --cuda 
### Single-cell pancreas data params
    5way-task
    Test runs = 500
    Test batch_size = 10
    LR = 0.0001
    Training subset = 15
    Step_size, Gamma = 100000, 0.5

