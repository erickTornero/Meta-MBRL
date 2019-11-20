import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import count
import random

class Trainer_ML:
    def __init__(self, meta_network, batch_sz, nepochs, split_ratio, inner_lr, outer_lr, device, M, K):
        self.meta_network   =   meta_network
        self.batch_size     =   batch_sz
        self.nepochs        =   nepochs
        self.split_ratio    =   split_ratio
        self.inner_lr       =   inner_lr
        self.outer_lr       =   outer_lr
        self.device         =   device
        self.M_points       =   M
        self.K_points       =   K

        self.optimizers     =   meta_network.optimizers
        self.ft_networks    =   meta_network.ft_networks
        self.dynamics_ml    =   meta_network.dynamics_ML
    
    def fine_tuning(self, X_data, target):
        assert X_data.ndim == 3, 'Dimension of data input must of 3 (tasks, nsamples, dimstates)'
        assert X_data.shape[0] == len(self.ft_networks)
        """ Initialize each network with respecto to metalearner """
        self.meta_network._copy_from_ML()
        


        for _i_net, _finetuning_net, optimizer in zip(count(), self.ft_networks, self.optimizers):
            """ Compute normalization of data """
            self.dynamics_ml.compute_normalization_stats(X_data[_i_net])
            data_normalized     =   self.normalize(X_data[_i_net])
            data_meta_train, data_meta_test, target_meta_train, target_meta_test    =   self.process_meta_data(data_normalized, target[_i_net])
            datazip =   list(zip(data_meta_train, target_meta_train))
            random.shuffle(datazip)
            x_train, y_train    =   zip(*datazip)
            #x_train             =   np.concatenate(x_train, axis=0)
            #y_train             =   np.concatenate(y_train, axis=0)
            #x_train, x_test, y_train, y_test =   train_test_split(X_data[_i_net], target[_i_net], test_size=self.split_ratio, random_state=42, shuffle=True)
            #x_train, x_test, y_train, y_test =   train_test_split(X_data[_i_net], target[_i_net], test_size=self.split_ratio, random_state=42, shuffle=True)

            #""" Compute normalization of data """
            #self.dynamics_ml.compute_normalization_stats(X_data[_i_net])
            #x_train =   self.normalize(x_train)
            #x_test  =   self.normalize(x_test)

            n_batches       =   len(y_train)//self.batch_size
            n_batches_test  =   len(target_meta_test)//self.batch_size if y_test.shape[0] >= self.batch_size else 1

            for n_epoch in range(self.nepochs):
                self.index      =   0
                for _ in range(n_batches):
                    x_batch     =   np.concatenate(x_train[self.index:self.index + self.batch_size, :], axis=0)  
                    y_batch     =   np.concatenate(y_train[self.index:self.index + self.batch_size, :], axis=0)

                    self.index  +=  self.batch_size

                    X_tensor    =   torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                    Y_tensor    =   torch.tensor(y_batch, dtype=torch.float32, device=self.device)

                    Y_pred      =   _finetuning_net(X_tensor)

                    """ Training steps """
                    optimizer.zero_grad()
                    output      =   torch.mean(torch.sum((Y_tensor - Y_pred)**2, dim=1))
                    output.backward()
                    optimizer.step()

                self.index      =   0
                for _ in range(n_batches_test):
                    x_batch_t   =   x_test[self.index:self.index + self.batch_size, :]
                    y_batch_t   =   y_test[self.index:self.index + self.batch_size, :]
                    self.index  +=  self.batch_size

                    X_tensor_test   =   torch.tensor(x_batch_t, dtype=torch.float32, device=self.device)
                    Y_tensor_test   =   torch.tensor(y_batch_t, dtype=torch.float32, device=self.device)

                    with torch.no_grad():
                        Y_pred_test =   _finetuning_net(X_tensor_test)
                        output_test =   torch.mean(torch.sum((Y_tensor_test - Y_pred_test)**2, dim=1))

    
    def normalize(self, x_input):
        """ 
            Normalization of data 
            We will normalization just by the dynamics of metalearner
        """
        
        assert self.dynamics_ml.mean_input is not None
        x   =   (x_input - self.dynamics_ml.mean_input)/(self.dynamics_ml.std_input + self.dynamics_ml.epsilon)
        return x
    
    def process_meta_data(self, x_data, y_target):
        data_meta_train     =   []
        data_meta_test      =   []
        target_meta_train   =   []
        target_meta_test    =   []
        len_data    =   y_target.shape[0]
        assert self.M_points + self.K_points < len_data, 'Too less data'

        for t in range(self.M_points, len_data - self.K_points):
            data_meta_train.append(x_data[t-self.M_points:t])
            data_meta_test.append(x_data[t:t+self.K_points])
            target_meta_train.append(y_target[t-self.M_points:t])
            target_meta_test.append(y_target[t:t+self.K_points])
        
        return data_meta_train, data_meta_test, target_meta_train, target_meta_test

