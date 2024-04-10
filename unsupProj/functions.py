'''
    Functions used for loading data, model and extracting features
    
    2022 Peggy Bevan
'''

import os
#import argparse
import yaml
import glob
#import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import umap as ump
import sklearn.cluster as cluster
from sklearn import metrics, datasets
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score

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
        output = {'features': [], 'img_path': []}
        for idx, data in enumerate(dataLoader): #if needed, adapt dataloader for prediction (no labels) 
            array = data[0]
            array = array.to(cfg['device'])
            features = model(array)
            features = features.cpu().detach().numpy() #bring back to cpu and convert to numpy
            filepath = data[1]
            output['features'].append(features)
            output['img_path'].append(filepath)
    output['features'] = np.concatenate((output['features']), axis = 0) #group batches into one large array
    output['img_path'] = np.concatenate((output['img_path']), axis = 0)
    return output

#finding optimal number of clusters using silhouette score
#kcomp is a dataframe with numimgs, nspecies, and a column called 'OK_dim' for each dimension
#dimensions is a list with integers - e.g. [2048]
#vecs & specs by site is a dictionary with list of vectors and labels in the same order - 
##keys = camera trap sites, values = feature vector arrays or species list
def findk(kcomp, dimensions, vecsbysite, specsbysite, kmax=15):
    for index,row in kcomp.iterrows():
        site = row['site']
        print(site)
        x = vecsbysite[site]
        y = specsbysite[site]
        kcomp['numimgs'][index] = len(x)
        kcomp['nspecies'][index] = len(set(y))
        #finding optimal value of k using the silhouette coefficient
        #if number of samples is less than kmax, change optimK
        for dim in dimensions:
            sil = []
            print(f'Finding optimal k for site {site} with {dim} dimensions')
            if len(x) > 5: #the 5 is arbitrary
                embedding = ump.UMAP(init = 'random', n_components=dim).fit_transform(x)
                if len(x) <= kmax:
                    kmax = len(x)-1
                    for k in range(2, kmax+1):
                        print(f'k = {k}')
                        kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
                        labels = kmeans.labels_ #underscore is needed
                        sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
                    #find index of max silhouette and add 2 (0 = 2) 
                    optimk = sil.index(max(sil))+2
                else:
                    for k in range(2, kmax+1):
                        print(f'k = {k}')
                        kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
                        labels = kmeans.labels_ #underscore is needed
                        sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
                        #find index of max silhouette and add 2 (0 = 2)
                    optimk = sil.index(max(sil))+2
            else:
                optimk = 'NA'
            column = ('OK_' + str(dim))
            kcomp[column][index] = optimk
            print(f'Optimal cluster number = {optimk}')
	

''' 
    if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
'''