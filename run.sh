# if running for the first time, activate conda environment
conda env create -f environment.yml
conda activate cv4e-unsupproj

# create text files of training and test image paths
python data_setup/ListTrainTest.py


#to call a model
python unsupProj/predict.py configs/cfgresnet50.yaml data/train.txt PegNet50 bct
#on my laptop each model takes about 3 minutes to run, except regnet128.

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt efficientNet2 bct

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt regnet128 bct

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train.txt convnextL bct

# now run on images of wild animals only

python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt PegNet50 bct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt efficientNet2 bct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt regnet128 bct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/train_wild.txt convnextL bct --wild

# new dataset: kenya (masaai mara camera traps)
python unsupProj/predict.py configs/fg_resnet50.yaml data/kenya/train.txt PegNet50 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt efficientNet2 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt regnet128 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt convnextL mmct

python unsupProj/predict.py configs/fg_resnet50.yaml data/kenya/train_wild.txt PegNet50 mmct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train_wild.txt efficientNet2 mmct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train_wild.txt regnet128 mmct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train_wild.txt convnextL mmct --wild
