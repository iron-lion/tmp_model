import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-nclass','--class_number',
                        type = int,
                        default = 5,
                        help='n-way setup, default: 5-way 5-shot')
    
    parser.add_argument('-nexample','--example_number',
                        type = int,
                        default = 5,
                        help='k-shot setup, default: 5-way 5-shot')
    
    parser.add_argument('-nbatch','--batch_size',
                        type = int,
                        default = 10,
                        help='training batch_size')
       
    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=200000)

    parser.add_argument('-nepoch_test', '--epochs_test',
                        type=int,
                        help='number of epochs to testing for',
                        default=500)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20) 

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
     
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')
    
    parser.add_argument('--save',
                        action='store_true',
                        help='training network saving')

    parser.add_argument('--early_stop1',
                        action='store_true',
                        help='training stop when batch_log mean accuracy > 0.99')

    parser.add_argument('-logb','--batch_log',
                        type = int,
                        help = 'print batch accuracy, every N episodes',
                        default=5000)
    
    parser.add_argument('-nsave','--save_at',
                        type = int,
                        help = 'save network status every N episodes',
                        default=50000)
    
    parser.add_argument('-fe','--fe_filename',
                        type = str,
                        default = 'networks/feature_encoder',
                        help='filename for feature encoder network')
    
    parser.add_argument('-rn','--rn_filename',
                        type = str,
                        default = 'networks/relation_network',
                        help='filename for relation network')

    parser.add_argument('-sp_sp', '--split_sample',
                        type=int,
                        help='Use small subset of data (~ number of sample)',
                        default=15)

    parser.add_argument('-sp_cl', '--split_class',
                        type=int,
                        help='Use small subset of data (~ number of class)',
                        default=10)
  
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=3)
        
    return parser

