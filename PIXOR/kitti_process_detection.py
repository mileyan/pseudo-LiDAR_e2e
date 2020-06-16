# The goal of this code:
# generate a bird-eye-view label image: "0" don't care; "1" non-car; "2" car
# generate bird-eye-view feature maps

from data.kitti_object import *
import torch
import numpy as np
from data.kitti_util import roty
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    box3d_roi_inds = in_hull(pc[:, 0: 3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def get_global_grid(zsize, xsize):
    z = np.linspace(0.0 + 70.0 / zsize / 2, 70.0 - 70.0 / zsize / 2, num=zsize)
    x = np.linspace(-40.0 + 80.0 / xsize / 2, 40.0 -
                    80.0 / xsize / 2, num=xsize)
    pc_grid = np.zeros(((zsize * xsize), 3))
    for i in range(zsize):
        for j in range(xsize):
            pc_grid[i * xsize + j, :] = [x[j], 0, z[i]]
    return pc_grid


def random_select(pts, rate=1/20):
    return pts[np.random.choice(pts.shape[0], int(pts.shape[0] * rate), replace=False)]


def get_3D_global_grid_extended(zsize, xsize, ysize):
    z = torch.linspace(0.0 - 70.0 / zsize / 2, 70.0 +
                       70.0 / zsize / 2, steps=zsize+2)
    x = torch.linspace(-40.0 - 80.0 / xsize / 2, 40.0 +
                       80.0 / xsize / 2, steps=xsize+2)
    y = torch.linspace(-1.0 - 3.5 / ysize / 2, 2.5 +
                       3.5 / ysize / 2, steps=ysize+2)
    pc_grid = torch.zeros((ysize+2, zsize+2, xsize+2, 3), dtype=torch.float)
    pc_grid[:, :, :, 0] = x.reshape(1, 1, -1)
    pc_grid[:, :, :, 1] = y.reshape(-1, 1, 1)
    pc_grid[:, :, :, 2] = z.reshape(1, -1, 1)
    return pc_grid

def gen_feature_diffused_tensor(pc_rect, feature_z, feature_x, grid_3D_extended,
                                diffused=False):
    valid_inds = (pc_rect[:, 2] < 70) & \
                 (pc_rect[:, 2] >= 0) & \
                 (pc_rect[:, 0] < 40) & \
                 (pc_rect[:, 0] >= -40) & \
                 (pc_rect[:, 1] < 2.5) & \
                 (pc_rect[:, 1] >= -1)
    pc_rect = pc_rect[valid_inds]

    pc_rect_quantized = torch.floor(
        pc_rect[:, :3] / 0.1).long().detach()
    pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
        + feature_x / 2
    pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + 10


    pc_rect_quantized += 1
    pc_rect_quantized = pc_rect_quantized.cuda()

    pc_rect_quantized_unique, inverse_idx = torch.unique(pc_rect_quantized, dim=0, return_inverse=True)

    pc_rect_assign = torch.exp(-((
        grid_3D_extended[
            pc_rect_quantized[:, 1],
            pc_rect_quantized[:, 2],
            pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01)

    pc_rect_assign_unique = scatter_mean(pc_rect_assign, inverse_idx)

    BEV_feature = torch.zeros(
        (35+2, feature_z+2, feature_x+2), dtype=torch.float).cuda()
    BEV_feature[pc_rect_quantized_unique[:, 1],
                pc_rect_quantized_unique[:, 2],
                pc_rect_quantized_unique[:, 0]] = pc_rect_assign_unique

    if diffused:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    else:
                        pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] + dx
                        pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + dy
                        pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] + dz

                        pc_rect_quantized_unique, inverse_idx = torch.unique(
                            pc_rect_quantized, dim=0, return_inverse=True)

                        pc_rect_assign = torch.exp(-((
                            grid_3D_extended[
                                pc_rect_quantized[:, 1],
                                pc_rect_quantized[:, 2],
                                pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01) / 26

                        pc_rect_assign_unique = scatter_mean(
                            pc_rect_assign, inverse_idx)

                        BEV_feature[pc_rect_quantized_unique[:, 1],
                                    pc_rect_quantized_unique[:, 2],
                                    pc_rect_quantized_unique[:, 0]] = \
                            BEV_feature[pc_rect_quantized_unique[:, 1],
                                        pc_rect_quantized_unique[:, 2],
                                        pc_rect_quantized_unique[:, 0]] + \
                                pc_rect_assign_unique

                        pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] - dx
                        pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] - dy
                        pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] - dz


    return BEV_feature[1:-1, 1:-1, 1:-1]



def get_car_grid_idx(obj, calib, label_grid, grid_idx, img_fov_inds_label=None):
    _, box3d_pts_3d = utils.compute_box_3d(
        obj, calib.P)  # 3D box corners in rect

    # set the height like this to make sure it includes some points in
    # our label_grid
    box3d_pts_3d[:4, 1] = -1
    box3d_pts_3d[4:, 1] = 1

    xmin, _, zmin = np.min(box3d_pts_3d, axis=0)
    xmax, _, zmax = np.max(box3d_pts_3d, axis=0)
    box3d_inds = (label_grid[:, 2] <= zmax) & \
                 (label_grid[:, 2] >= zmin) & \
                 (label_grid[:, 0] <= xmax) & \
                 (label_grid[:, 0] >= xmin)
    if img_fov_inds_label is not None:
        box3d_inds = box3d_inds & img_fov_inds_label
    idx_in_box3d = grid_idx[box3d_inds]
    pc_in_box3d = label_grid[idx_in_box3d, :]

    _, inds = extract_pc_in_box3d(pc_in_box3d, box3d_pts_3d)
    return idx_in_box3d[inds]

def shift_objects(objects, t, calib):
    R = roty(t)
    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        obj.ry += t
        center = R.dot(np.array(obj.t))
        obj.t = (center[0], center[1], center[2])
    return objects

def shift_pc(pc, t):
    R = roty(t)
    pc[:, :3] = pc[:, :3].dot(R.T)
    return pc

def get_features(dataset, data_idx, feature_grid, feature_z, feature_x,
                 rfttype='max', shift=0, diffused=False, grid_3D_extended=None):
    calib = dataset.get_calibration(data_idx)

    pc_velo = dataset.get_lidar(data_idx)

    # project lidar points to camera space
    pc_velo[:, :3] = calib.project_velo_to_rect(pc_velo[:, :3])
    pc_rect = pc_velo

    img = dataset.get_image(data_idx)
    img_height, img_width, img_channel = img.shape

    _, _, valid_inds_fov = get_rect_in_image_fov(
        pc_rect[:, :3], calib, 0, 0, img_width,
        img_height, True, clip_distance=2.0)
    pc_rect = pc_rect[valid_inds_fov]

    if not np.isclose(shift, 0):
        pc_rect = shift_pc(pc_rect, shift)

    if diffused:
        BEV_feature = gen_feature_diffused(
            pc_rect, feature_z, feature_x, grid_3D_extended)
        BEV_feature_reflex = None
    else:
        BEV_feature, BEV_feature_reflex = gen_feature(pc_rect, rfttype,
                                                    feature_z, feature_x)
        BEV_feature, BEV_feature_reflex = torch.from_numpy(BEV_feature).float(), \
            torch.from_numpy(BEV_feature_reflex).float()

    return BEV_feature, BEV_feature_reflex

def get_labels(dataset, data_idx, label_grid, label_z, label_x, shift=0,
    type_whitelist=('Car', )):

    calib = dataset.get_calibration(data_idx)
    img = dataset.get_image(data_idx)
    img_height, img_width, img_channel = img.shape
    class_label = np.zeros(label_grid.shape[0], dtype=np.int32)
    reg_label = np.zeros((6, label_grid.shape[0]), dtype=np.float32)
    grid_idx = np.arange(label_grid.shape[0])
    _, _, img_fov_inds_label = get_rect_in_image_fov(
        label_grid, calib, 0, 0, img_width, img_height, True, clip_distance=2.0)


    objects = dataset.get_label_objects(data_idx)
    if not np.isclose(shift, 0):
        objects = shift_objects(objects, shift, calib)

    for obj_idx in range(len(objects)):
        if objects[obj_idx].type not in type_whitelist:
            continue

        obj = objects[obj_idx]
        theta = obj.ry
        w, l = obj.w, obj.l
        center = np.array(obj.t)
        car_grid_idx = get_car_grid_idx(
            obj, calib, label_grid, grid_idx, img_fov_inds_label=img_fov_inds_label)
        class_label[car_grid_idx] = 1  # valid car

        for idx in car_grid_idx:
            reg_label[0, idx] = np.cos(theta)
            reg_label[1, idx] = np.sin(theta)
            reg_label[2, idx] = center[0] - label_grid[idx][0]  # dx
            reg_label[3, idx] = center[2] - label_grid[idx][2]  # dz
            reg_label[4, idx] = np.log(w)
            reg_label[5, idx] = np.log(l)

    class_label = np.reshape(class_label, (label_z, label_x))
    reg_label = np.reshape(reg_label, (6, label_z, label_x))

    return torch.from_numpy(class_label), torch.from_numpy(reg_label)
