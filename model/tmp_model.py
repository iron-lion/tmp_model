import glob
import os
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch
from torch.optim.lr_scheduler import StepLR
from model import rl_model as L2C
from model import common as common
from sklearn.metrics.cluster import adjusted_rand_score

class TMP:
    def __init__(self, params, c_len, e_dim_1 = 4000, e_dim_2 = 2000, e_dim_3 = 1000, r_dim_1 = 500, r_dim_2 = 100):
        
        self.channel_len = c_len
        self.embedding_first = e_dim_1
        self.embedding_second = e_dim_2
        self.embedding_third = e_dim_3
        self.relation_first = r_dim_1
        self.relation_second = r_dim_2
        self.feature_dim = e_dim_3

        self.device = params.device
        self.epochs = params.epochs
        self.epochs_test = params.epochs_test
        self.lr = params.learning_rate
        self.lrS = params.lr_scheduler_step
        self.lrG = params.lr_scheduler_gamma
        self.class_number = params.class_number
        self.example_number = params.example_number
        self.batch_size = params.batch_size
        self.feature_file_name = params.fe_filename
        self.relation_file_name = params.rn_filename
        
        self.init_model(c_len, e_dim_1, e_dim_2, e_dim_3, r_dim_1, r_dim_2, params.device)
        self.load_model(self.feature_file_name, self.relation_file_name)

        self.save = params.save
        self.early_stop1 = params.early_stop1
        self.batch_log = params.batch_log
        self.save_at = params.save_at


    def init_model(self, c_len, e_dim_1, e_dim_2, e_dim_3, r_dim_1, r_dim_2, device):
        self.feature_encoder = L2C.GEPEncoder(c_len, e_dim_1, e_dim_2, e_dim_3).to(device)
        self.relation_network = L2C.RelationNetwork(e_dim_3, r_dim_1, r_dim_2).to(device)

    
    def load_model(self, fn_name, rn_name):
        if os.path.exists(fn_name):
            self.feature_file_name = fn_name
            self.feature_encoder.load_state_dict(torch.load(fn_name, map_location=torch.device(self.device)))
            print("load feature encoder success!", fn_name)
        else:
            print("fail to load feature encoder")
            #exit('EXIT: cannot find feature encoder', fn_name)
        if os.path.exists(rn_name):
            self.relation_file_name = rn_name
            self.relation_network.load_state_dict(torch.load(rn_name, map_location=torch.device(self.device)))
            print("\nload relation network success!", rn_name)
        else:
            print("fail to load relation network")
            #exit('EXIT: cannot find relation network', rn_name)


    def init_optim(self, fe_param, rn_param, learning_rate):
        fe_optim = torch.optim.Adam(params=fe_param, lr=learning_rate)
        rn_optim = torch.optim.Adam(params=rn_param, lr=learning_rate)
        
        return fe_optim, rn_optim


    def set_filenames(self, fe_file, rn_file):
        self.feature_file_name = fe_file
        self.relation_file_name = rn_file


    def test_with_histogram(self, train_data_dic, gene_set, sorting, file_name):
        test_accs = []
        test_aris = []

        histogram = pd.DataFrame(columns=train_data_dic.keys(), index=train_data_dic.keys())
        histogram = histogram.replace(np.nan,0)

        for epoch in range(self.epochs_test):
            # class balance loader
            samples, sample_labels, batches, batch_labels, label_converter \
                = common.sample_test_split(train_data_dic, self.class_number, self.example_number, \
                                            self.batch_size, gene_set, sorting)
            samples = torch.Tensor(samples.transpose()).float()
            batches = torch.Tensor(batches.transpose()).float()
            batch_labels = torch.LongTensor(batch_labels.values)
            
            _, predict_labels = self.l2c_loss(samples, sample_labels, batches, batch_labels)
 
            rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(len(predict_labels))]
            acc = np.sum(rewards) / len(batch_labels)
            ari = adjusted_rand_score(predict_labels.cpu().detach().data, batch_labels)
  
            test_accs.append(acc)
            test_aris.append(ari)
            
            for pair in zip(batch_labels.data, predict_labels.cpu().detach().data):
                # print(label_converter[pair[0]], '->'  ,label_converter[pair[1].item()])
                histogram.at[label_converter[pair[0].item()], label_converter[pair[1].item()]] \
                    = histogram.at[label_converter[pair[0].item()], label_converter[pair[1].item()]] + 1

        test_accuracy,h = common.mean_confidence_interval(test_accs)
        ari_accuracy,ari_h = common.mean_confidence_interval(test_aris)

        print("test#","Cumulative accuracy:", "h:", "ARI:", "ARI_h:")
        print(self.epochs_test,test_accuracy,h, ari_accuracy, ari_h)
    
        print(histogram)
        compression_opts = dict(method='zip', archive_name=file_name+'.csv')
        histogram.to_csv(file_name+'.zip', compression=compression_opts)
        return 


    def test(self, train_data_dic, gene_set, sorting):
        test_accs = []
        test_aris = []
        for epoch in range(self.epochs_test):
            # class balance loader
            samples, sample_labels, batches, batch_labels, label_converter \
                = common.sample_test_split(train_data_dic, self.class_number, self.example_number, \
                                            self.batch_size, gene_set, sorting)
            samples = torch.Tensor(samples.transpose()).float()
            batches = torch.Tensor(batches.transpose()).float()
            batch_labels = torch.LongTensor(batch_labels.values)
            
            _, predict_labels = self.l2c_loss(samples, sample_labels, batches, batch_labels)
 
            rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(len(predict_labels))]
            acc = np.sum(rewards) / len(batch_labels)
            ari = adjusted_rand_score(predict_labels.cpu().detach().data, batch_labels)
            
            test_accs.append(acc)
            test_aris.append(ari)
        test_accuracy,h = common.mean_confidence_interval(test_accs)
        ari_accuracy,ari_h = common.mean_confidence_interval(test_aris)
        print("test#","Cumulative accuracy:", "h:", "ARI:", "ARI_h:")
        print(self.epochs_test,test_accuracy,h, ari_accuracy, ari_h)
        
        return 


    def few_shot_test(self, test_data_dic, gene_set, sorting):
        self.test(test_data_dic, gene_set, sorting)


    
    def fix_test(self, example_data_dic, train_data_dic, gene_set, sorting):
        test_accs = []
        test_aris = []

        for j in range(self.epochs_test):
            samples, sample_labels, _, _, label_converter \
                = common.sample_test_split(example_data_dic, self.class_number, self.example_number, 0, gene_set, sorting)
            samples = torch.Tensor(samples.transpose()).float()
            for epoch in range(5):
                # class balance loader
                _, _, batches, batch_labels, _ \
                    = common.sample_test_split(train_data_dic, self.class_number, 0, \
                                                self.batch_size, gene_set, sorting, label_converter)

                batches = torch.Tensor(batches.transpose()).float()
                batch_labels = torch.LongTensor(batch_labels.values)
                
                _, predict_labels = self.l2c_loss(samples, sample_labels, batches, batch_labels)
     
                rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(len(predict_labels))]
                acc = np.sum(rewards) / len(batch_labels)
                ari = adjusted_rand_score(predict_labels.cpu().detach().data, batch_labels)
                
                test_accs.append(acc)
                test_aris.append(ari)
        test_accuracy,h = common.mean_confidence_interval(test_accs)
        ari_accuracy,ari_h = common.mean_confidence_interval(test_aris)
        print("test#","Cumulative accuracy:", "h:", "ARI:", "ARI_h:")
        print(self.epochs_test,test_accuracy,h, ari_accuracy, ari_h)
        
        return 


    def train(self, train_data_dic, gene_set, sorting):
        print("> init training...")
        batch_accs = []

        feature_encoder_optim, relation_network_optim = self.init_optim(self.feature_encoder.parameters(),
                                                                    self.relation_network.parameters(),
                                                                    self.lr)
        feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=self.lrS,gamma=self.lrG) # decay LR
        relation_network_scheduler = StepLR(relation_network_optim,step_size=self.lrS,gamma=self.lrG) # decay LR

        """       
        for par in self.feature_encoder.parameters():
            par.requires_grad = False
        """

        for epoch in range(self.epochs):
            feature_encoder_optim.step()
            relation_network_optim.step()
           
            # class balance loader
            samples, sample_labels, batches, batch_labels, label_converter \
                = common.sample_test_split(train_data_dic, self.class_number, self.example_number, \
                                            self.batch_size, gene_set, sorting)
            samples = torch.Tensor(samples.transpose()).float()
            batches = torch.Tensor(batches.transpose()).float()
            batch_labels = torch.LongTensor(batch_labels.values)
            
            loss, predict_labels = self.l2c_loss(samples, sample_labels, batches, batch_labels)
 
            rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(len(predict_labels))]
            batch_acc = np.sum(rewards) / len(batch_labels)
            test_ari = adjusted_rand_score(predict_labels.cpu().detach().data, batch_labels)
 
            # training
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()
              
            loss.backward()
            batch_accs.append(batch_acc)

            feature_encoder_scheduler.step()
            relation_network_scheduler.step()

            # Non-negative encoder
            for par in self.feature_encoder.parameters():
                par.data.clamp_(0)

            if (epoch+1)%self.batch_log == 0:
                print("episode:",epoch+1," loss:",loss.data, " acc:", np.mean(batch_accs))
                if self.early_stop1 and (np.mean(batch_accs) > 0.99):
                    print("episode:",epoch+1, "early stop")
                    break
                batch_accs.clear()

            if self.save and (epoch+1)%self.save_at == 0:
                # save networks
                self.save_suffix = epoch
                self.network_save(self.feature_file_name, self.relation_file_name)
        if self.save:
            # save networks
            self.save_suffix = epoch
            self.network_save(self.feature_file_name, self.relation_file_name)

        return 


    def network_save(self, feature_file_name, relation_file_name):
        torch.save(self.feature_encoder.state_dict(),feature_file_name + "." + str(self.save_suffix))
        torch.save(self.relation_network.state_dict(),relation_file_name + "." + str(self.save_suffix))


    def l2c_loss(self, samples, sample_labels, batches, batch_labels):
        sample_features = self.feature_encoder(Variable(samples.float()).to(self.device))
        sample_features = sample_features.view(self.class_number, self.example_number, self.feature_dim)
        sample_features = torch.mean(sample_features,1).squeeze(1) 
        sample_features_ext = sample_features.unsqueeze(0).repeat(self.batch_size*self.class_number,1,1) 
        
        batch_features = self.feature_encoder(Variable(batches.float()).to(self.device))
        batch_features_ext = batch_features.unsqueeze(0).repeat(self.class_number,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,self.feature_dim*2)
        relations = self.relation_network(relation_pairs).view(-1,self.class_number)

        mse = nn.MSELoss().to(self.device)
        one_hot_labels = Variable(torch.zeros(self.batch_size*self.class_number, \
                                            self.class_number).scatter_(1, (batch_labels).view(-1,1), 1)).to(self.device)
        loss = mse(relations,one_hot_labels)
        _,predict_labels = torch.max(relations.data,1)
        
        return loss, predict_labels


    def intra_class_loss(self, samples, sample_labels, batches, batch_labels):
        sample_features = self.feature_encoder(Variable(training_images).to(self.device))
        sample_features_t = sample_features.view(self.class_number, self.example_number, self.feature_dim)
        sample_features_t = torch.transpose(sample_features_t,0,1)
        sample_features_t = sample_features_t.reshape(self.example_number, self.class_number, -1).squeeze(1)
        sample_features_t = sample_features_t.repeat(self.example_number, 1, 1)

        test_features_ext = sample_features.unsqueeze(0).repeat(self.class_number,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)
        mse = nn.MSELoss().to(self.device)
        return mse(test_features_ext,sample_features_t)



