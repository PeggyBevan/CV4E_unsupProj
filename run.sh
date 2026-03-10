# if running for the first time, activate conda environment
conda env create -f environment.yml
conda activate cv4e-unsupproj

# create text files of training and test image paths
python data_setup/ListTrainTest.py


#to call a model
python unsupProj/predict.py configs/cfgresnet50.yaml data/train.txt PegNet50
#on my laptop this takes about 3 minutes to run

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt efficientNet2 

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt regnet128

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt convnextL

# now run on images of wild animals only

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt PegNet50 --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt efficientNet2 --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt regnet128 --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt convnextL --wild