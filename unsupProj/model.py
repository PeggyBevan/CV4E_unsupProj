'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.
    2022 Peggy Bevan
'''

import torch.nn as nn
import torch
from torchvision.models import resnet

#Class name is whatever you want it to be. nn.Module inherits 
#properties from pre-made module
class CustomPegNet50(nn.Module):

    def __init__(self):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomPegNet50, self).__init__()

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
        #I don't care about prediction - just want to ouput features
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