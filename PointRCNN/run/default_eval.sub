#!/bin/bash
#SBATCH -J train                                   # Job name
#SBATCH -o logs/train.o%j                           # Name of stdout output file (%j expands to jobId)
#SBATCH -e logs/train.o%j                       # Name of stderr output file (%j expands to jobId)
#SBATCH -N 1                                                     # Total number of nodes requested
#SBATCH -n 4                                                      # Number of cores requested
#SBATCH --mem=32G                                     # Memory pool for all cores
#SBATCH -t 120:00:00                                          # Run time (hh:mm:ss)
#SBATCH --partition=kilian --gres=gpu:1080ti:2           # Which queue it should run on.
#SBATCH --nodelist=harpo

POINT_STYLE=default
NUM_GPUS=2
BATCH_SIZE=$((4 * $NUM_GPUS)) # upto 4 batches per GPU

module rm cuda cudnn
module add cuda/10.0 cudnn/v7.4-cuda-10.0
module list

#conda deactivate
#conda activate 3d

set -e

cd ../tools

# EVAL
# python eval_rcnn_depth.py --cfg_file cfgs/"$POINT_STYLE".yaml --rpn_ckpt ../output/rpn/pretrained/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rpn

python eval_rcnn_depth.py --cfg_file cfgs/"$POINT_STYLE".yaml --ckpt /home/rq49/PointRCNN_ORI/sdn_fix_sparse/rcnn/checkpoint_epoch_70.pth --batch_size 8 --eval_mode rcnn
