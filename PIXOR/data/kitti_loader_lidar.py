# -*- coding: utf-8 -*-

from __future__ import print_function

import PIL.Image as Image
import cv2
import numpy as np
import random
import os.path as osp
from torch.utils.data import Dataset
import torch
import pickle
from data import fusion
import torchvision.transforms as transforms

from kitti_process_detection import get_global_grid, get_labels, get_3D_global_grid_extended
from data.kitti_object import *
import torch.nn.functional as F


h, w      = 700, 800
train_h   = h
train_w   = w
val_h     = h
val_w     = w

label_map_ds_rate = 4
feature_map_zsize = 700
feature_map_xsize = 800
feature_map_ysize = 35

class KittiDataset_Fusion_stereo(Dataset):

    def __init__(self, txt_file, flip_rate=0., lidar_dir='lidar',
                 label_dir='label_2', calib_dir='calib', image_dir='image_2',
                 image2_dir='image_3',
                 random_shift_scale=0,
                 aug_noise=0, reduce_points_rate=1.0,
                 root_dir="/scratch/datasets/KITTI/object",
                 only_feature=False, split='training', image_downscale=1, crop_height=-1,
                 **kargs):

        self.data = [int(line.rstrip()) for line in open(txt_file)]

        self.flip_rate = flip_rate

        self.reduce_points_rate = reduce_points_rate
        self.aug_noise = aug_noise
        self.random_shift_scale = random_shift_scale
        self.only_feature = only_feature

        self.dataset = kitti_object(root_dir=root_dir, split=split,
                                    lidar_dir=lidar_dir, label_dir=label_dir,
                                    calib_dir=calib_dir, image_dir=image_dir,
                                    image2_dir=image2_dir)
        self.feature_z, self.feature_x = feature_map_zsize, feature_map_xsize
        self.label_z, self.label_x = self.feature_z // label_map_ds_rate, \
            self.feature_x // label_map_ds_rate
        self.label_grid = get_global_grid(self.label_z, self.label_x)
        self.feature_grid = get_global_grid(self.feature_z, self.feature_x)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.image_downscale = image_downscale
        self.crop_height = crop_height

        self.n_features = 35

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_idx = self.data[idx]
        shift = (np.random.rand()*2-1)*self.random_shift_scale/180*np.pi

        # image
        image = self.dataset.get_image1(data_idx)
        image = self.trans(image)[None, :, :, :]
        right_pad = 1248 - image.shape[-1]
        bottom_pad = 384 - image.shape[-2]
        image = F.pad(image, (0, right_pad, 0, bottom_pad), "constant", 0)
        if self.image_downscale > 1:
            image = F.interpolate(
                image, scale_factor=(1./self.image_downscale))

        image = image[0]

        depth_map = self.dataset.get_depth_map(data_idx)
        calib = self.dataset.get_calibration(data_idx)

        bev_index, img_index = fusion.Fusion(depth_map, calib, shift)

        f = calib.P[0, 0] * 0.54
        imgL = Image.open(self.dataset.get_image_path(
            data_idx, is_left=True)).convert('RGB')
        imgR = Image.open(self.dataset.get_image_path(
            data_idx, is_left=False)).convert('RGB')
        imgL = self.trans(imgL)
        imgR = self.trans(imgR)
        # pad to (384, 1248)
        # print(imgL.shape)
        C, H, W = imgL.shape
        top_pad = 384 - H
        right_pad = 1248 - W

        if self.crop_height < 0:
            imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
            imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
            depth_map = F.pad(torch.Tensor(depth_map),
                              (0, right_pad, top_pad, 0), "constant", -1)
            h_shift = 0
        else:
            h_shift = imgL.shape[1] - self.crop_height
            imgL = F.pad(imgL, (0, right_pad, 0, 0), "constant", 0)
            imgR = F.pad(imgR, (0, right_pad, 0, 0), "constant", 0)
            depth_map = F.pad(torch.Tensor(depth_map),
                              (0, right_pad, 0, 0), "constant", -1)
            imgL = imgL[:,-self.crop_height:,:]
            imgR = imgR[:,-self.crop_height:,:]
            depth_map = depth_map[-self.crop_height:,:]
            H = self.crop_height


        if not self.only_feature:
            class_label, reg_label = \
                get_labels(self.dataset, data_idx,
                           self.label_grid, self.label_z, self.label_x,
                           shift=shift,
                           type_whitelist=('Car',))
            class_label, reg_label = \
                class_label.float(), reg_label.float(), \

        flip = False
        if random.random() < self.flip_rate:
            flip = True
            image = torch.flip(image, [2])
            if not self.only_feature:
                class_label = torch.flip(class_label, [1])
                reg_label = torch.flip(reg_label, [2])
                reg_label[0, :, :] *= -1  # cos(pi - theta)
                reg_label[2, :, :] *= -1  # dx

        if self.only_feature:
            return {'imgL': imgL, 'imgR': imgR, 'f': f, 'depth_map': depth_map,
                    'idx': data_idx,
                    'image': image, 'img_index': img_index, 'bev_index': bev_index, 'h_shift':h_shift,
                    'ori_shape': [H, W], 'flip': flip, 'a_shift': shift}
        else:
            return {'imgL': imgL, 'imgR': imgR, 'f': f,  'depth_map': depth_map,
                    'cl': class_label, 'rl': reg_label, 'idx': data_idx,
                    'image': image, 'img_index': img_index, 'bev_index': bev_index, 'h_shift':h_shift,
                    'ori_shape': [H, W], 'flip': flip, 'a_shift': shift}
