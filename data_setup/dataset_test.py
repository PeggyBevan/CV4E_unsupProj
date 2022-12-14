dataset_test.py

#script to test if unsupProj/dataset.py is working
#2022 Peggy Bevan

#to import dataset, need to be in the right working directory
cd unsupProj/
import dataset, yaml, functions
from dataset import CTDataset
from functions import 


#load config file
cfg = yaml.safe_load(open('/home/pegbev/CV4E_unsupProj/configs/cfg_resnet50.yaml'))
#point at file directory
img_list_path = '/home/pegbev/data/train.txt'
#set dataset
dataset = CTDataset(cfg, img_list_path, None)
#test length function is working
len(dataset)

#see what one line looks like
dataset[0][1]
#unpacking a tuple

#test dataLoader 
img_list_path = '/home/pegbev/data/train.txt'
cfg = cfg = yaml.safe_load(open('/home/pegbev/CV4E_unsupProj/configs/cfg_resnet50.yaml'))
dl = create_dataloader(cfg, img_list_path)