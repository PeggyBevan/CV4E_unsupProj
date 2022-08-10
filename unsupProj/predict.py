'''
    Model Run and 'predict' model out put
    Here, model and datset at loaded and run across images 
    2022 Peggy Bevan
'''

#load other scripts - must be in the same directory
import model, dataset
from model import CustomPegNet50
from dataset import CTDataset



#create model and apply parameters
model = CustomPegNet50()
#this might cause an error if no GPU
model.cuda

# to call the fn
img_list_path = '/home/pegbev/data/train.txt'
cfg = 
dl = create_dataloader(cfg, img_list_path)
#model = load_model(ckpt_path)
prediction_dict = predict(cfg, dl, model)

#write as json into filepath 


