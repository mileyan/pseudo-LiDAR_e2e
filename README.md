# End-to-end Pseudo-LiDAR for Image-Based 3D Object Detection
This paper has been accepted by Computer Vision and Pattern Recognition 2020.

[End-to-end Pseudo-LiDAR for Image-Based 3D Object Detection](https://arxiv.org/abs/2004.03080)

by [Rui Qian*](https://rui1996.github.io/), [Divyansh Garg*](http://divyanshgarg.com/), [Yan Wang*](https://www.cs.cornell.edu/~yanwang/), [Yurong You*](http://yurongyou.com/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Bharath Hariharan](http://home.bharathh.info/), [Mark Campbell](https://campbell.mae.cornell.edu/), [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/) and [Wei-Lun Chao](http://www-scf.usc.edu/~weilunc/)

### Citation
```
@inproceedings{qian2020end,
  title={End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection},
  author={Qian, Rui and Garg, Divyansh and Wang, Yan and You, Yurong and Belongie, Serge and Hariharan, Bharath and Campbell, Mark and Weinberger, Kilian Q and Chao, Wei-Lun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5881--5890},
  year={2020}
}
```

###Abstract

Reliable and accurate 3D object detection is a necessity for safe autonomous driving. Although LiDAR sensors can provide accurate 3D point cloud estimates of the environment, they are also prohibitively expensive for many settings. Recently, the introduction of pseudo-LiDAR (PL) has led to a drastic reduction in the accuracy gap between methods based on LiDAR sensors and those based on cheap stereo cameras. PL combines state-of-the-art deep neural networks for 3D depth estimation with those for 3D object detection by converting 2D depth map outputs to 3D point cloud inputs. However, so far these two networks have to be trained separately. In this paper, we introduce a new framework based on differentiable Change of Representation (CoR) modules that allow the entire PL pipeline to be trained end-to-end. The resulting framework is compatible with most state-of-the-art networks for both tasks and in combination with PointRCNN improves over PL consistently across all benchmarks --- yielding the highest entry on the KITTI image-based 3D object detection leaderboard at the time of submission. 

### Contents

```
Root
    | PIXOR
    | PointRCNN
```

We provide end-to-end modification on pointcloud-based detector(PointRCNN) and voxel-based detector(PIXOR).

The PIXOR folder contains implementation of **Quantization** as described in Section3.1 of the paper. Also it contains our own implementation of [PIXOR](https://arxiv.org/abs/1902.06326). 

The PointRCNN folder contains implementation of **Subsampling** as described in Section3.2 of the paper. It is developed based on the [codebase](https://github.com/sshaoshuai/PointRCNN) of [Shaoshuai Shi](https://github.com/sshaoshuai).

### Data Preparation 
This repo is based on the KITTI dataset. Please download it and prepare the data as same as in [Pseudo-LiDAR++](https://github.com/mileyan/Pseudo_Lidar_V2#pseudo-lidar-accurate-depth-for-3d-object-detection-in-autonomous-driving). Please refer to its readme for more details.

### Training and evaluation
Please refer to each subfolder for details.

### Questions
This repo is currently maintained by Rui Qian and Yurong You. Please feel free to ask any question.

You can reach us by put an issue or email:
[rq49@cornell.edu](mailto:rq49@cornell.edu?subject=[GitHub]%20Pseudo-Lidar_E2E),
[yy785@cornell.edu](mailto:yy785@cornell.edu?subject=[GitHub]%20Pseudo-Lidar_E2E)