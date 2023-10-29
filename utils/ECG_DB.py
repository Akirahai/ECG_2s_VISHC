import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig

def r_peaks(sig):
    
    return sig

class ECG_DB(Dataset):
    def __init__(self, phase, path_db, path_lb, folds):
        super(ECG_DB, self).__init__()
        self.phase=phase
        lb=pd.read_csv(path_lb)
        lb = lb[lb['fold'].isin(folds)]
        
        self.path_db=path_db
        self.labels = lb
        self.classes = ['SNR','AF', 'IAVB', 'LBBB', 'RBBB','STD']       
        self.data_dict = {}
        self.label_dict = {}
        
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        id = row['id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.path_db,id))
        ecg_data = transform(ecg_data, self.phase == 'train')
        arr_ecg, _ = ecg_data.shape
        ecg_data = ecg_data[-2000:, ]
        result = np.zeros((2000, 12))
        result[-arr_ecg:, :] = ecg_data
        
        if self.label_dict.get(id):
            labels = self.label_dict.get(id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[id] = labels
            
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()