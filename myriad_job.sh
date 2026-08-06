#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=06:00:00                # Time limit (hh:mm:ss) - adjust as needed
#$ -l memory=8G                    # RAM per core
#$ -l gpu=1                        # Request 1 GPU (crucial for PyTorch models)
#$ -ac app=pytorch                 # Ensures PyTorch GPU environment compatibility
#$ -cwd                            # Run job from current working directory
#$ -N predict_kenya                # Job name shown in queue
#$ -j y                            # Merge standard error and standard output into one log file
#$ -o output_predict_$JOB_ID.log   # Log file name

# 1. Load required modules (if applicable on Myriad) or activate your conda env
# Replace 'unsup' with your exact conda environment name if different:
source ~/.bashrc
conda activate unsup

# 2. Print job details to log for debugging
echo "Job started on $(date) on host $(hostname)"
echo "Using GPU device: $CUDA_VISIBLE_DEVICES"

# 3. Execute your prediction commands sequentially
echo "Starting predictions..."

python unsupProj/predict.py configs/fg_resnet50.yaml data/kenya/train.txt PegNet50 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt efficientNet2 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt regnet128 mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train.txt convnextL mmct
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train_wild.txt regnet128 mmct --wild
python unsupProj/predict.py configs/cfg_resnet50.yaml data/kenya/train_wild.txt convnextL mmct --wild

echo "All predictions completed successfully at $(date)"