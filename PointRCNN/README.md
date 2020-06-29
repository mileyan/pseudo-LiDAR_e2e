# PointRCNN-PL-end2end

This folder contains code to reproduce the results on end-to-end training of PointRCNN model on stereo 3D object detection.

## Dependency
* pytorch 1.1.0 (torchvision 0.3.0)
* [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
* opencv-python
* tqdm
* numpy
* scipy

## Pre-train models
Download PRCNN model pretrained on SDN on KITTI [here](https://drive.google.com/file/d/1aIP1kwhrtlkyV59_9iOl8261ACgGhLup/view?usp=sharing)  and pretrained depth model [here](https://drive.google.com/file/d/1qlvZPezFsnEWDNNHT9cGpmxIY388k4iS/view?usp=sharing).

After downloading model weights, create `depth_network/results/stack_finetune_from_sceneflow` folder and put the depth weight under the folder.

## Install
Please refer to the original [PointRCNN](https://github.com/sshaoshuai/PointRCNN) repo for detailed install instruction. 

## Run PointRCNN_PL_end2end training and evaluation

run end2end training (on 2 TITAN RTX GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_rcnn_depth.py --gt_database "" --cfg_file cfgs/e2e.yaml \
 --batch_size 4 --train_mode end2end --ckpt_save_interval 1 --ckpt <detection_model_pretrain_path> \
--epochs 10 --mgpus --finetune
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 python eval_rcnn_depth.py --cfg_file cfgs/e2e.yaml \
--depth_ckpt <end2end_trained_depth_model_path>  --ckpt <end2end_trained_detector_model_path> \
--batch_size 4 --eval_mode rcnn --mgpus 
```