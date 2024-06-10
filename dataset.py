import torch
import os
import pandas as pd
from sklearn import preprocessing

class feature_dataset(torch.utils.data.Dataset):
     def __init__(self,data_path, preprocess = None):
          assert os.path.exists(data_path)
          self.rawfeature_data = pd.read_csv(data_path,index_col=0)

          if preprocess == None:
              standarized = preprocessing.StandardScaler()
              print('rescale not mentioned, defaulting to standard scale type')
          elif preprocess == 'MinMax':
              standarized = preprocessing.MinMaxScaler()
          self.processedfeature_data = standarized.fit_transform(self.rawfeature_data)
        #   numpy_data = self.processedfeature_data.to_numpy()

# Convert NumPy array to PyTorch tensor and ensure it's a float tensor
          self.processedfeature_tensordata = torch.tensor(self.processedfeature_data, dtype=torch.float)

     def __len__(self):
        return len(self.processedfeature_data)
        # return len(self.all_label)

     def __getitem__(self, idx):
         return self.processedfeature_tensordata[idx]

# Datasetloader for training, testing and validate dataset 

class Datasetloader:
    def __init__(self, data_path, batch_size=32, train_split=0.7, val_split=0.1, test_split=0.2, random_seed=123):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_seed

        self.ds = feature_dataset(data_path = self.data_path)

        ##split into train, test and val
        self.df_train, self.df_val, self.df_test = self.train_val_test_split()

        ##dataloader
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.df_train, batch_size= self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=self.df_val, batch_size= self.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.df_test, batch_size= self.batch_size, shuffle=True)

    def train_val_test_split(self):
        ds_size = len(self.ds)
        train_size = int(self.train_split * ds_size)
        val_size = int(self.val_split * ds_size)
        test_size = ds_size - train_size - val_size
        return torch.utils.data.random_split(self.ds, [train_size, val_size,test_size])
    
    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    
# Loader = Datasetloader(data_path='/home/schnablelab/Documents/GenePrediction/MaizePanGeneCount_SorghumCount.csv')

# tr, vl, test= Loader.get_dataloaders()
# from architecture import Autoencoder
# for batch, data in enumerate(tr):
#     print(batch, data)