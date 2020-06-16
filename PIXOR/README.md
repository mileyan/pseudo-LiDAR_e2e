# PIXOR-PL-end2end

This folder contains code to reproduce the results on end-to-end training of PIXOR model on stereo 3D object detection.

## Dependency
* pytorch 1.1.0 (torchvision 0.3.0)
* [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
* configargparse
* opencv-python
* tqdm
* numpy
* scipy

## Pre-train models
Download pixor model pretrained on SDN on KITTI [here](https://drive.google.com/open?id=1ptaxgElztlC83vRKvwTQ5VwUBBk1We7X) and pretrained depth model [here](https://drive.google.com/open?id=1WtEDA13_qHuFyeljkx4QwAgZ_2Jkf8ez).

## Run PIXOR_PL_end2end finetuneing

compile the kitti offical evaluation code:
```bash
cd evaluation/kitti_eval
./compile.sh
```

run end2end finetuning (on 2 TITAN RTX GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_pixor_e2e.py -c configs/fusion.cfg --depth_pretrain <depth_model_pretrain_path> --pixor_pretrain <detection_model_pretrain_path>
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_pixor_e2e.py -c configs/fusion.cfg --mode eval --eval_ckpt <full_model_checkpoint_path>
```