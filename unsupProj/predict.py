'''
    Model Run and 'predict' model out put
    Here, model and datset at loaded and run across images 
    2022 Peggy Bevan
'''

print('Loading packages')
#load other scripts - must be in the same directory (unsupProj)
import model, dataset, functions
import yaml, json
#import pickle
from model import CustomPegNet50
from model import SwavNet
from dataset import CTDataset
from functions import create_dataloader, predict
import torch
import numpy as np


print('Running PegNet50')
#create model and apply parameters
PegNet = CustomPegNet50()
#this might cause an error if no GPU
PegNet = PegNet.cuda()


# to call the fn
img_list_path = '/home/pegbev/data/train.txt'
cfg = yaml.safe_load(open('/home/pegbev/CV4E_unsupProj/configs/cfg_resnet50.yaml'))
#until GPU is bigger, num workers must be 2
#cfg['num_workers'] = 2
#cfg['device'] = 'cpu'
dl = create_dataloader(cfg, img_list_path)

print('Creating feature vectors...')
prediction_dict = predict(cfg, dl, PegNet)

vectors = prediction_dict['features']
vecs = np.concatenate((vectors), axis = 0)
np.save("../../output/PegNet_fvect_norm.npy", vecs) #norm = with normalize transform

img_path = prediction_dict['img_path']
#convert to numpy array by concatenating
imgs = np.concatenate((imgs), axis = 0)
np.save("../../output/PegNet_imgvect_norm.npy", imgs) #norm = with normalize transform

#remove model from GPU
PegNet.cpu()

#Run SwavNet
SwavNet = SwavNet()
SwavNet = SwavNet.cuda()

print('Creating feature vectors...')
prediction_dict = predict(cfg, dl, SwavNet)

vectors = prediction_dict['features']
vecs = np.concatenate((vectors), axis = 0)
np.save("../../output/Swav_fvect.npy", vecs)

img_path = prediction_dict['img_path']
#convert to numpy array by concatenating
imgs = np.concatenate((imgs), axis = 0)
np.save("../../output/Swav_imgvect.npy", imgs)