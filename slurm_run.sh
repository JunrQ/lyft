#!/bin/sh
#SBATCH --job-name=TestJob
#SBATCH --output=test.out
#SBATCH --error=testError.err
#SBATCH --qos=q_msai
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=128G


conda activate py36-torch

# MY_GIT=/mnt/nvme/zcq/git
# MY_GIT=/home/junr/git
MY_GIT=/home/MSAI/zhou0365/git

gpus=0
export CUDA_VISIBLE_DEVICES=${gpus}

config=v0.0.2.config

python run.py \
  --config ${MY_GIT}/lyft/config/${config} \
  --dir ${MY_GIT}/lyft/log/${config} \
  --log-frequence 100 \
  --save-frequence 1000 \
  --eval-frequence 2000 \
  --resume true

