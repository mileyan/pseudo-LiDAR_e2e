import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch_scatter import scatter_max

import math
import random
import numpy as np
from depth_network import logger
import os
import shutil
from depth_network.models import *
import kitti_util
import batch_utils

from PIL import Image
from tensorboardX import SummaryWriter
import ipdb


def loader(path):
    return Image.open(path).convert('RGB')


def dynamic_baseline(calib):
    P3 = calib.P3
    P = calib.P2
    baseline = P3[0, 3] / (-P3[0, 0]) - P[0, 3] / (-P[0, 0])
    return baseline


class DepthModel():
    def __init__(self, maxdisp, down, maxdepth, pretrain, save_tag, mode='TRAIN', dynamic_bs=False,
                     lr=0.001, mgpus=False, lr_stepsize=[10, 20], lr_gamma=0.1):

        result_dir = os.path.join('../', 'output', 'depth', save_tag)
        # set logger
        log = logger.setup_logger(os.path.join(result_dir, 'training.log'))

        # set tensorboard
        writer = SummaryWriter(result_dir + '/tensorboardx')

        model = stackhourglass(maxdisp, down=down, maxdepth=maxdepth)

        # Number of parameters
        log.info('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        if mgpus or mode == 'TEST':
            model = nn.DataParallel(model)
        model = model.cuda()

        torch.backends.cudnn.benchmark = True

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = MultiStepLR(
            optimizer, milestones=lr_stepsize, gamma=lr_gamma)

        if pretrain is not None:
            if os.path.isfile(pretrain):
                log.info("=> loading pretrain '{}'".format(pretrain))
                checkpoint = torch.load(pretrain)
                if mgpus or mode == 'TEST':
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(self.strip_prefix(checkpoint['state_dict']))
                optimizer.load_state_dict(checkpoint['optimizer'])

            else:
                log.info(
                    '[Attention]: Do not find checkpoint {}'.format(pretrain))

        optimizer.param_groups[0]['lr'] = lr

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = model
        self.dynamic_bs = dynamic_bs
        self.mode = mode
        self.result_dir = result_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def load_data(self, batch_left_img, batch_right_img, batch_gt_depth, batch_calib):
        left_imgs, right_imgs, calibs = [], [], []
        for left_img, right_img, calib in zip(
                batch_left_img, batch_right_img, batch_calib):
            if self.dynamic_bs:
                calib = calib.P2[0, 0] * dynamic_baseline(calib)
            else:
                calib = calib.P2[0, 0] * 0.54

            calib = torch.tensor(calib)
            left_img = self.img_transform(left_img)
            right_img = self.img_transform(right_img)

            # pad to (384, 1248)
            C, H, W = left_img.shape
            top_pad = 384 - H
            right_pad = 1248 - W
            left_img = F.pad(
                left_img, (0, right_pad, top_pad, 0), "constant", 0)
            right_img = F.pad(
                right_img, (0, right_pad, top_pad, 0), "constant", 0)

            left_imgs.append(left_img)
            right_imgs.append(right_img)
            calibs.append(calib)

        left_img = torch.stack(left_imgs)
        right_img = torch.stack(right_imgs)
        calib = torch.stack(calibs)

        gt_depth = torch.from_numpy(batch_gt_depth).cuda(non_blocking=True)

        return left_img.float(), right_img.float(), gt_depth.float(), calib.float()

    def train(self, batch, start=2.0, max_high=1.0):
        imgL, imgR, gt_depth, calib = self.load_data(
            batch['left_image'], batch['right_image'], batch['gt_depth'], batch['calib'])
        imgL, imgR, gt_depth, calib = imgL.cuda(), imgR.cuda(), gt_depth.cuda(), calib.cuda()

        # ---------
        mask = (gt_depth >= 1) * (gt_depth <= 80)
        mask.detach_()
        #print('mask', torch.sum(mask).float()/(mask.size()[0]*mask.size()[1]*mask.size()[2]))
        # ----

        output1, output2, output3 = self.net(imgL, imgR, calib)
        output3 = torch.squeeze(output3, 1)

        def hook_fn(grad):
            print(grad.size())
            a = (grad == 0).float()
            rate = 100 * torch.sum(a) / (grad.size()[0] * grad.size()[1] * grad.size()[2])
            print('depth_map', rate, torch.mean(grad)/(rate/100), torch.max(grad), torch.min(grad))
            print('one_norm', torch.sum(torch.abs(grad)))

        loss = 0.5 * F.smooth_l1_loss(output1[mask], gt_depth[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], gt_depth[mask], size_average=True) + F.smooth_l1_loss(output3[mask], gt_depth[mask],
                                                                                size_average=True)

        points = []
        for depth, calib_info, image, sample_id in zip(
                output3, batch['calib'], batch['left_image'], batch['sample_id']):
            calib_info = kitti_util.Calib(calib_info)
            W, H = image.size
            depth = depth[-H:, :W]
            cloud = depth_to_pcl(calib_info, depth, max_high=max_high)
            cloud = filter_cloud(cloud, image, calib_info)
            cloud = transform(cloud, calib_info, sparse_type='angular_min', start=2.0)
            # save_pcl(cloud, 'points/sparse_{}'.format(sample_id))
            points.append(cloud)

        det_batch = batch_utils.get_detector_batch(points, batch, mode='TRAIN')
        return loss, det_batch

    def eval(self, batch, max_high=1.0):
        imgL, imgR, gt_depth, calib = self.load_data(
            batch['left_image'], batch['right_image'], batch['gt_depth'], batch['calib'])
        imgL, imgR, gt_depth, calib = imgL.cuda(), imgR.cuda(), gt_depth.cuda(), calib.cuda()

        # ---------
        mask = (gt_depth >= 1) * (gt_depth <= 80)
        mask.detach_()
        #print('mask', torch.sum(mask).float() / (mask.size()[0] * mask.size()[1] * mask.size()[2]))
        # ----
        
        with torch.no_grad():
            output3 = self.net(imgL, imgR, calib)
            output3 = torch.squeeze(output3, 1)
            #loss = F.smooth_l1_loss(output3[mask], gt_depth[mask], size_average=True)
            loss = 0

            points = []
            for depth, calib_info, image, sample_id in zip(
                    output3, batch['calib'], batch['left_image'], batch['sample_id']):
                calib_info = kitti_util.Calib(calib_info)
                W, H = image.size

                depth = depth[-H:, :W]
                cloud = depth_to_pcl(calib_info, depth, max_high=max_high)
                cloud = filter_cloud(cloud, image, calib_info)
                cloud = transform(cloud, calib_info, sparse_type='angular_min', start=2.0)
                points.append(cloud)

            det_batch = batch_utils.get_detector_batch(points, batch, mode='TEST')

        return loss, det_batch

    def save_checkpoint(self, epoch, is_best=False, filename='checkpoint.pth.tar'):
        save_dir = os.path.join(self.result_dir, 'ckpt')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_RMSE = 0  # TODO: Add RMSE loss
        state = {
            'epoch': epoch + 1,
            'arch': 'stackhourglass',
            'state_dict': self.net.state_dict(),
            'best_RMSE': best_RMSE,
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, save_dir + '/' + filename)
        if is_best:
            shutil.copyfile(save_dir + '/' + filename,
                            save_dir + '/model_best.pth.tar')

        #shutil.copyfile(save_dir + '/' + filename, save_dir +
        #                '/checkpoint_{}.pth.tar'.format(epoch+1))

    def strip_prefix(self, state_dict, prefix='module.'):
        if not all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
        return stripped_state_dict


def depth_to_pcl(calib, depth, max_high=1.):
    rows, cols = depth.shape
    c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
    points = torch.stack([c.t(), r.t(), depth], dim=0)
    points = points.reshape((3, -1))
    points = points.t()
    cloud = calib.img_to_lidar(points[:, 0], points[:, 1], points[:, 2])
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    lidar = cloud[valid]

    # pad 1 in the intensity dimension
    lidar = torch.cat(
        [lidar, torch.ones((lidar.shape[0], 1), device='cuda')], 1)
    lidar = lidar.float()
    return lidar


def transform(points, calib_info, sparse_type, start=2.):
    if sparse_type == 'angular':
        points = random_sparse_angular(points)
    if sparse_type == 'angular_min':
        points = nearest_sparse_angular(points, start)
    if sparse_type == 'angular_numpy':
        points = points.cpu().numpy()
        points = pto_ang_map(points).astype(np.float32)
        points = torch.from_numpy(points).cuda()

    return points


def filter_cloud(velo_points, image, calib):
    W, H = image.size
    _, _, valid_inds_fov = get_lidar_in_image_fov(
        velo_points[:, :3], calib, 0, 0, W, H, True)
    velo_points = velo_points[valid_inds_fov]

    # depth, width, height
    valid_inds = (velo_points[:, 0] < 120) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1.5) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]
    return velo_points


def gen_ang_map(velo_points, start=2., H=64, W=512, device='cuda'):
    dtheta = math.radians(0.4 * 64.0 / H)
    dphi = math.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:,
                                    1], velo_points[:, 2], velo_points[:, 3]

    d = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = torch.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = math.radians(45.) - torch.asin(y / r)
    phi_ = (phi / dphi).long()
    phi_ = torch.clamp(phi_, 0, W - 1)

    theta = math.radians(start) - torch.asin(z / d)
    theta_ = (theta / dtheta).long()
    theta_ = torch.clamp(theta_, 0, H - 1)
    return [theta_, phi_]


def random_sparse_angular(velo_points, H=64, W=512, slice=1, device='cuda'):
    """
    :param velo_points: Pointcloud of size [N, 4]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, H=64, W=512, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)

    depth_map = depth_map
    velo_points = velo_points
    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    theta_, phi_ = theta_, phi_

    # Currently, does random subsample (maybe keep the points with min distance)
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map.cuda()

    depth_map = depth_map[0:: slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    return depth_map[depth_map[:, 0] != -1.0]



def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

#   np.random.shuffle(velo_points)
    dtheta = np.radians(0.4 * 3.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_] = velo_points

    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def nearest_sparse_angular(velo_points, start=2., H=64, W=512, slice=1, device='cuda'):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, start, H, W, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)
    depth_map = min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda')
    # depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    sparse_points = depth_map[depth_map[:, 0] != -1.0]
    return sparse_points


def min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda'):
    N = velo_points.shape[0]

    idx = theta_ * W + phi_  # phi_ in range [0, W-1]
    depth = torch.arange(0, N, device='cuda')

    sampled_depth, argmin = scatter_max(depth, idx)
    mask = argmin[argmin != -1]
    return velo_points[mask]


def save_pcl(point_cloud, path='point'):
    point_cloud = point_cloud.detach().cpu()
    np.save(path, point_cloud)


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d, pts_rect_depth = calib.lidar_to_img(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo
