import sklearn.preprocessing
import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.utils import shuffle
import sklearn


class BarlowtwinsTrainer:

    def __init__(self, **kwargs):
        ## Reuired files for below functions

        self.dataset_path = kwargs['dataset_path']
        self.random_seed = kwargs['random_seed']
        self.prop_split = kwargs['prop_split']
        self.raw_data=pd.read_csv(self.dataset_path, index_col=0).dropna()
        
        self.train_data, self.validation_data = self.train_test_split()

        self.df, self.train_data, self.validation_data = self.dataPreProcess()

        # self.BarlowTwin = self.model()
    
    def train_test_split(self):

        # train_label = []
        # validation_labels = []
        # train_indices = []
        # validation_indices = []

        train_data, validation_data = sklearn.train_test_split(self.raw_data, test_size=1-self.prop_split, random_state=self.random_seed)

        return train_data, validation_data
    
    def dataPreProcess(self):

        df = self.raw_data
        standarized = sklearn.preprocessing.StandardScaler()
        df_train = standarized.fit_transform(self.train_data)
        df_validation = standarized.fit_transform(self.validation_data)

        return df, df_train, df_validation
    


class architecture(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.linear1=torch.nn.Linear(kwargs["input_shape"], 2200)
        self.activation=torch.nn.SELU()
        self.linear2=torch.nn.Linear(2200, 3000)
        self.dropout=torch.nn.Dropout(0.3)
        self.linear3=torch.nn.Linear(3000, 2024)
        self.linear4=torch.nn.Linear(2024, kwargs["output_shape"])

        
        self.decodelinear1=torch.nn.Linear(kwargs["output_shape"], 2024)
        self.decodelinear2=torch.nn.Linear(2024, 3000)
        self.decodelinear3=torch.nn.Linear(3000, 2200)
        self.decodelinear4=torch.nn.Linear(2200, kwargs["input_shape"])
        self.tanactivation=torch.nn.Tanh()
        
        
    def forward(self, x):
        conv2D_layer1 = self.conv1(x)
        # # print(conv2D_layer1.shape)
        flatten = conv2D_layer1.view(-1)
        # print(flatten.shape)
        encodelinear1=self.linear1(flatten)

        ##encoder
        encodelinear1=self.activation(encodelinear1)

        encodelinear2=self.linear2(encodelinear1)
        encodelinear2=self.activation(encodelinear2)

        encodelinear3=self.linear3(encodelinear2)
        encodelinear3=self.activation(encodelinear3)
        encodelinear3=self.dropout(encodelinear3)

        encodelinear4=self.linear4(encodelinear3)

        ##decoder
        decodelinear1=F.selu(self.decodelinear1(encodelinear4))
        decodelinear2=F.selu(self.decodelinear2(decodelinear1))
        decodelinear3=F.selu(self.decodelinear3(decodelinear2))
        decodelinear4=F.selu(self.decodelinear4(decodelinear3))
        # print(decodelinear4.shape)
        
        return flatten, decodelinear4

















    
