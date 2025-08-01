import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

class OpenFWI_Dataset(TensorDataset):
    def __init__(self, data_base_path, transform=None,dtype=torch.float32):
        self.data_base_path = data_base_path
        self.transform = transform

        # get all the velocity models
        self.data_path_list = [os.path.join(self.data_base_path, file) for file in os.listdir(self.data_base_path)]

        # retrive all the velocity model data
        self.all_velocity_models = []
        for data_path in self.data_path_list:
            vel_model_subsets = self.get_velocity_subset(data_path)
            self.all_velocity_models.extend(vel_model_subsets)
        
        self.all_velocity_models = np.array(self.all_velocity_models)
        self.all_velocity_models = torch.tensor(self.all_velocity_models,dtype=dtype).unsqueeze(1)

        # get the number of velocity models
        self.num_velocity_models = len(self.all_velocity_models)
    
    def normalize(self, data):
        """
        Normalize the data to [0,1]
        """
        # [batch_size, w, h]
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return data
    
    def crop_velocity_model(self, data):
        """
        Crop the velocity model to [64,64]
        """
        data = data[:,3:67,:64]
        return data

    def get_velocity_subset(self, data_path):
        data = np.load(data_path)
        data = data.squeeze()
        data = self.crop_velocity_model(data)
        data = self.normalize(data)
        return data

    def __len__(self):
        return self.num_velocity_models
    
    def __getitem__(self, idx):
        return self.all_velocity_models[idx]