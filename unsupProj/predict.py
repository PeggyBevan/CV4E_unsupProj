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
print('Model loaded successfully...')

# to call the fn
img_list_path = '/home/pegbev/data/train.txt'
cfg = yaml.safe_load(open('/home/pegbev/CV4E_unsupProj/configs/cfg_resnet50.yaml'))
#until GPU is bigger, num workers must be 2
#cfg['num_workers'] = 2
#cfg['device'] = 'cpu'
dl = create_dataloader(cfg, img_list_path)

print('Creating feature vectors...')
prediction_dict = predict(cfg, dl, PegNet)
print('Prediction complete')
vectors = prediction_dict['features']
np.save("../../output/PegNet_fvect_norm.npy", vectors) #norm = with normalize transform

img_path = prediction_dict['img_path']
#convert to numpy array by concatenating
np.save("../../output/PegNet_imgvect_norm.npy", img_path) #norm = with normalize transform
print('Embeddings saved')
#remove model from GPU
PegNet.cpu()

#Run SwavNet
SwavNet = SwavNet()
SwavNet = SwavNet.cuda()
print('Model loaded successfully...')
print('Creating feature vectors...')
prediction_dict = predict(cfg, dl, SwavNet)
print('Prediction complete')
vectors = prediction_dict['features']
np.save("../../output/Swav_fvect.npy", vectors)

img_path = prediction_dict['img_path']
np.save("../../output/Swav_imgvect.npy", img_path)
print('Embeddings saved')
SwavNet.cpu()


#Run Omi's EmbModel
print('Running EmbNet')
checkpoint = torch.load('data/kenya_resnet50_simclr_2022_05_05__16_34_13.pt')
args = checkpoint['args']

EmbNet = EmbModel(eval(args['backbone']), args).to(args['device'])
msg = EmbNet.load_state_dict(checkpoint['state_dict'], strict=True)
EmbNet = EmbNet.cuda()

print('Model loaded successfully...')
prediction_dict = predict(cfg, dl, EmbModel)
print('Prediction complete')
vectors = prediction_dict['features']
np.save("../../output/Emb_fvect.npy", vectors)

img_path = prediction_dict['img_path']
np.save("../../output/Emb_imgvect.npy", img_path)
print('Embeddings saved')

