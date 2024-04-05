#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, 2 is the max
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=1:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N RN50_predict

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucbtbev/Scratch/CV4E_unsupProj/
# Change into temporary directory to run work
cd $TMPDIR

# load the cuda module (in case you are running a CUDA program)
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

pip3 install yaml json pytorch numpy os PIL

# Run the application
python unsupProj/predict.py
