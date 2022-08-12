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
from dataset import CTDataset
from functions import create_dataloader, predict


#create model and apply parameters
model = CustomPegNet50()
#this might cause an error if no GPU
model.cuda


# to call the fn
img_list_path = '/home/pegbev/data/train.txt'
cfg = yaml.safe_load(open('/home/pegbev/CV4E_unsupProj/configs/cfg_resnet50.yaml'))
#until GPU is bigger, num workers must be 2
#cfg['num_workers'] = 2
#cfg['device'] = 'cpu'
dl = create_dataloader(cfg, img_list_path)

print('Creating feature vectors...')
prediction_dict = predict(cfg, dl, model)

#write as json into filepath 
print('Saving dictionary to json')
with open("../main_output.json", "w") as outfile:
    json.dump(prediction_dict, outfile)