'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.
    2022 Peggy Bevan
'''

import torch
import torch.nn as nn
from torchvision.models import resnet

#Class name is whatever you want it to be. nn.Module inherits 
#properties from pre-made module
class CustomPegNet50(nn.Module):

    def __init__(self):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomPegNet50, self).__init__() #calls the 'super' class (nn.Module)

        self.feature_extractor = resnet.resnet50(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # remove the very last layer from the original, 1000-class output
        # and replace with an identity block 
        # removing softmax and linear model
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...


    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-50 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        #I don't care about prediction - just want to output features
        #prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return features

class SwavNet(nn.Module):

    def __init__(self):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(SwavNet, self).__init__()

        self.feature_extractor = torch.hub.load('facebookresearch/swav:main', 'resnet50') # "pretrained": use weights pre-trained on ImageNet

        # remove the very last layer from the original, 1000-class output
        # and replace with an identity block 
        # removing softmax and linear model
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc  # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features    # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()  # discard last layer...


    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-50 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        #I don't care about prediction - just want to ouput features
        #prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return features

class EmbModel(nn.Module):
    
    def __init__(self, base_encoder, args):
        super().__init__()
    
        self.enc = base_encoder(pretrained=args['pretrained'])
        self.feature_dim = self.enc.fc.in_features
        self.projection_dim = args['projection_dim'] 
        self.proj_hidden = 512

        self.simsiam = False
        self.embed_context = False
        

        if args['dataset'] == 'cifar_mnist':
            # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
            # See Section B.9 of SimCLR paper.
            self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.enc.maxpool = nn.Identity()

        # remove final fully connected layer of the backbone
        self.enc.fc = nn.Identity()  

        if args['store_embeddings']:
            self.emb_memory = torch.zeros(args['num_train'], args['projection_dim'], 
                                          requires_grad=False, device=args['device'])
            
        if args['train_loss'] == 'simsiam': 
            self.simsiam = True
            # combination of standard simclr projector with BN and simiam predictor
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                           nn.BatchNorm1d(self.proj_hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.proj_hidden, self.projection_dim)) 
            self.predictor = PredictionMLP(self.projection_dim, self.projection_dim//2, self.projection_dim)
        
        else:
            
            # standard simclr projector
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.proj_hidden, self.projection_dim)) 
            
        
    def update_memory(self, inds, x):
        m = 0.9
        with torch.no_grad():
            self.emb_memory[inds] = m*self.emb_memory[inds] + (1.0-m)*F.normalize(x.detach().clone(), dim=1, p=2)
            self.emb_memory[inds] = F.normalize(self.emb_memory[inds], dim=1, p=2)        
        
    def forward(self, x, 
    #only_feats=False, 
    context=None
    ):
        op = self.enc(x) 
        '''
        if not only_feats:
        
            if self.simsiam:
                op['emb'] = self.projector(op['feat'])
                op['emb_p'] = self.predictor(op['emb'])
        
            else:
                op['emb'] = self.projector(op['feat'])
        '''

        return op
