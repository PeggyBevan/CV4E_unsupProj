'''
    Model Run and 'predict' model out put
    Here, model and datset at loaded and run across images 
    2022 Peggy Bevan
'''
print('Loading packages')

#load other scripts - must be in the same directory (unsupProj)
import model, dataset, functions
import yaml
#import pickle
from model import PegNet50
from model import SwavNet
from model import EmbModel
from model import efficientNet2, regnet128, convnextL
from dataset import CTDataset
from model import NP_RN50_full
from functions import create_dataloader, predict
import torch
import numpy as np
#from torchvision.models import resnet18, resnet50
import torch.nn as nn
import typer
import os

# note: this script uses typer. any unknown variables are taken in 
# when the script is run. For example, to run PegNet50, use the command:
# python predict.py configs/cfg_resnet50.yaml data/train.txt PegNet50
# the unknown variables are the config file, the image list file, and the model name.
# there is also a setting for wild or not wild. If wild = true, this will be added to the file name.


def create_model(model_name):
    '''
    Create a model based on the model name.
    '''
    if model_name == 'PegNet50':
        return PegNet50()
    elif model_name == 'SwavNet':
        return SwavNet()
    elif model_name == 'EmbModel':
        return EmbModel()
    elif model_name == 'NP_RN50_full':
        return NP_RN50_full()
    elif model_name == 'efficientNet2':
        return efficientNet2()
    elif model_name == 'regnet128':
        return regnet128()
    elif model_name == 'convnextL':
        return convnextL()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def predict_model (cfg_path, img_list_path, model_name, wild: bool = False):
    '''
    Run the model on the images in the dataset and return the feature vectors.
    '''
    print(f'Running {model_name}')
    #create model and apply parameters
    model = create_model(model_name)
    #this might cause an error if no GPU
    #PegNet = PegNet.cuda()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    print(f'{model_name} loaded successfully...')

    # to call the fn
    cfg = yaml.safe_load(open(cfg_path))
    dl = create_dataloader(cfg, img_list_path)

    print('Creating feature vectors...')
    prediction_dict = predict(cfg, dl, model)
    print('Prediction complete')
    vectors = prediction_dict['features']
    #if output dir doesn't exist, create it
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    if wild: #add wild to file name if wild = true
        np.save(f"output/{model_name}_wild_fvect_norm.npy", vectors) #norm = with normalize transform
    else:
        np.save(f"output/{model_name}_fvect_norm.npy", vectors) #norm = with normalize transform

    img_path = prediction_dict['img_path']
    #convert to numpy array by concatenating
    if wild: #add wild to file name if wild = true
        np.save(f"output/{model_name}_wild_imgvect_norm.npy", img_path) #norm = with normalize transform
    else:
        np.save(f"output/{model_name}_imgvect_norm.npy", img_path) #norm = with normalize transform
    print('Embeddings saved')
    


if __name__ ==  '__main__':
    
    typer.run(predict_model)
    #predict_model('configs/cfg_resnet50.yaml', 'data/train.txt', 'PegNet50')

#     print('Running PegNet50')
#     #create model and apply parameters
#     PegNet = CustomPegNet50()
#     #this might cause an error if no GPU
#     #PegNet = PegNet.cuda()
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     elif torch.backends.mps.is_available():
#         device = torch.device('mps')
#     else:
#         device = torch.device('cpu')
#     PegNet = PegNet.to(device)

#     print('PegNet Model loaded successfully...')

#     # to call the fn
#     img_list_path = 'data/train.txt'
#     cfg = yaml.safe_load(open('configs/cfg_resnet50.yaml'))
#     #until GPU is bigger, num workers must be 2
#     #cfg['num_workers'] = 2
#     #cfg['device'] = 'cpu'
#     dl = create_dataloader(cfg, img_list_path)

#     print('Creating feature vectors...')
#     prediction_dict = predict(cfg, dl, PegNet)
#     print('Prediction complete')
#     vectors = prediction_dict['features']
#     np.save("output/PegNet_fvect_norm.npy", vectors) #norm = with normalize transform

#     img_path = prediction_dict['img_path']
#     #convert to numpy array by concatenating
#     np.save("output/PegNet_imgvect_norm.npy", img_path) #norm = with normalize transform
#     print('Embeddings saved')
#     #remove model from GPU
#     PegNet.cpu()

#     # #Run SwavNet
#     # SwavNet = SwavNet()
#     # SwavNet = SwavNet.cuda()
#     # print('SwavNet loaded successfully...')
#     # print('Creating feature vectors...')
#     # prediction_dict = predict(cfg, dl, SwavNet)
#     # print('Prediction complete')
#     # vectors = prediction_dict['features']
#     # np.save("output/Swav_fvect.npy", vectors)

#     # img_path = prediction_dict['img_path']
#     # np.save("output/Swav_imgvect.npy", img_path)
#     # print('Embeddings saved')
#     # SwavNet.cpu()


#     # #Run Omi's EmbModel
#     # print('Running EmbNet')
#     # checkpoint = torch.load('../../data/kenya_resnet50_simclr_2022_05_05__16_34_13.pt')
#     # args = checkpoint['args']

#     # EmbNet = EmbModel(eval(args['backbone']), args).to(args['device'])
#     # msg = EmbNet.load_state_dict(checkpoint['state_dict'], strict=True)
#     # EmbNet = EmbNet.cuda()

#     # print('Model loaded successfully...')
#     # prediction_dict = predict(cfg, dl, EmbNet)
#     # print('Prediction complete')
#     # vectors = prediction_dict['features']
#     # np.save("../../output/Emb_fvect.npy", vectors)

#     # img_path = prediction_dict['img_path']
#     # np.save("../../output/Emb_imgvect.npy", img_path)
#     # print('Embeddings saved')

# #Run ResNet50 fully trained model
#     # print('Running RN_50_full')
    
    
#     # checkpoint = torch.load('data/nepal_resnet50_supervised_2022_10_21__13_16_12.pt', map_location=torch.device('cpu'))
#     # args = checkpoint['args']

#     # base_encoder = eval(args['backbone'])
#     # RN_50 = base_encoder(weights=None)
#     # RN_50.fc = nn.Identity()

#     #  #RN_50 = NP_RN50_full(eval(args['backbone']), args).to(device)
#     # msg = RN_50.load_state_dict(checkpoint['state_dict'], strict=False)

#     # if torch.cuda.is_available():
#     #     device = torch.device('cuda')
#     # else:
#     #     device = torch.device('cpu')
#     #     RN_50 = RN_50.to(device)
  
#     # print('Model loaded successfully...')
#     # prediction_dict = predict(cfg, dl, RN_50)
#     # print('Prediction complete')
#     # vectors = prediction_dict['features']
#     # np.save("../output/NP_RN50_full_fvect.npy", vectors)

#     # img_path = prediction_dict['img_path']
#     # np.save("../output/NP_RN50_full_imgvect.npy", img_path)
#     # print('RN50 full Embeddings saved')
