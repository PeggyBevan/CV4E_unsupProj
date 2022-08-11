'''
    Functions used for loading data, model and extracting features
    
    2022 Peggy Bevan
'''

import os
#import argparse
import yaml
import glob
#from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from dataset import CTDataset
from model import CustomPegNet50


def create_dataloader(cfg, img_list_path):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''

    dataset_instance = CTDataset(cfg, img_list_path, None) # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers']
        )
    return dataLoader
'''
I don't think this is needed for now.
def load_model(cfg):
'''
        #Creates a model instance and loads the latest model state weights.
'''
    model_instance = CustomPegNet50()  # create an object instance of our CustomResNet18 class
    return model_instance
'''
def predict(cfg, dataLoader, model):
    '''
        loads in data and model together, and extracts features for each row
        Output is a dictionary with feacture vector and img_path for each item
    '''
    with torch.no_grad():   # no gradients needed for prediction
        output = {}
        for idx, data in enumerate(dataLoader): #if needed, adapt dataloader for prediction (no labels) 
            array = data[0]
            array = array.to(device)
            features = model(array) #adapt model fn to return what you want
            filepath = data[1]
            output[idx] = {'features': features, 'img_path': filepath}
            break
    return output
	



''' 
    if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
'''