'''
    Model Run and 'predict' model out put
    Here, model and datset at loaded and run across images 
    2022 Peggy Bevan
'''

if __name__ ==  '__main__':
    print('Loading packages')

    #load other scripts - must be in the same directory (unsupProj)
    import model, dataset, functions
    import yaml
    #import pickle
    from model import CustomPegNet50
    from model import SwavNet
    from model import EmbModel
    from dataset import CTDataset
    from model import NP_RN50_full
    from functions import create_dataloader, predict
    import torch
    import numpy as np
    from torchvision.models import resnet18, resnet50
    import torch.nn as nn



    print('Running PegNet50')
    #create model and apply parameters
    PegNet = CustomPegNet50()
    #this might cause an error if no GPU
    #PegNet = PegNet.cuda()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        PegNet = PegNet.to(device)

    print('PegNet Model loaded successfully...')

    # to call the fn
    img_list_path = 'data/train.txt'
    cfg = yaml.safe_load(open('configs/cfg_resnet50.yaml'))
    #until GPU is bigger, num workers must be 2
    #cfg['num_workers'] = 2
    #cfg['device'] = 'cpu'
    dl = create_dataloader(cfg, img_list_path)

    # print('Creating feature vectors...')
    # prediction_dict = predict(cfg, dl, PegNet)
    # print('Prediction complete')
    # vectors = prediction_dict['features']
    # np.save("output/PegNet_fvect_norm.npy", vectors) #norm = with normalize transform

    # img_path = prediction_dict['img_path']
    # #convert to numpy array by concatenating
    # np.save("output/PegNet_imgvect_norm.npy", img_path) #norm = with normalize transform
    # print('Embeddings saved')
    # #remove model from GPU
    # PegNet.cpu()

    # #Run SwavNet
    # SwavNet = SwavNet()
    # SwavNet = SwavNet.cuda()
    # print('SwavNet loaded successfully...')
    # print('Creating feature vectors...')
    # prediction_dict = predict(cfg, dl, SwavNet)
    # print('Prediction complete')
    # vectors = prediction_dict['features']
    # np.save("output/Swav_fvect.npy", vectors)

    # img_path = prediction_dict['img_path']
    # np.save("output/Swav_imgvect.npy", img_path)
    # print('Embeddings saved')
    # SwavNet.cpu()


    # #Run Omi's EmbModel
    # print('Running EmbNet')
    # checkpoint = torch.load('../../data/kenya_resnet50_simclr_2022_05_05__16_34_13.pt')
    # args = checkpoint['args']

    # EmbNet = EmbModel(eval(args['backbone']), args).to(args['device'])
    # msg = EmbNet.load_state_dict(checkpoint['state_dict'], strict=True)
    # EmbNet = EmbNet.cuda()

    # print('Model loaded successfully...')
    # prediction_dict = predict(cfg, dl, EmbNet)
    # print('Prediction complete')
    # vectors = prediction_dict['features']
    # np.save("../../output/Emb_fvect.npy", vectors)

    # img_path = prediction_dict['img_path']
    # np.save("../../output/Emb_imgvect.npy", img_path)
    # print('Embeddings saved')

#Run ResNet50 fully trained model
    print('Running RN_50_full')
    
    base_encoder = eval(args['backbone'])
    RN_50 = base_encoder(weights=None)
    RN_50.fc = nn.Identity()

    checkpoint = torch.load('data/nepal_resnet50_supervised_2022_10_21__13_16_12.pt', map_location=torch.device('cpu'))
    args = checkpoint['args']
     #RN_50 = NP_RN50_full(eval(args['backbone']), args).to(device)
    msg = RN_50.load_state_dict(checkpoint['state_dict'], strict=False)

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    #     RN_50 = RN_50.to(device)
  
    print('Model loaded successfully...')
    prediction_dict = predict(cfg, dl, RN_50)
    print('Prediction complete')
    vectors = prediction_dict['features']
    np.save("../output/NP_RN50_full_fvect.npy", vectors)

    img_path = prediction_dict['img_path']
    np.save("../output/NP_RN50_full_imgvect.npy", img_path)
    print('RN50 full Embeddings saved')
