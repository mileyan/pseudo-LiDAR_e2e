import numpy as np
import torch


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def Fusion(depth_map, calib, t):
    high, width = depth_map.shape
    u = torch.arange(width)
    v = torch.arange(high)
    index = torch.stack(torch.meshgrid((u, v))).permute((2, 1, 0)).numpy()
    depth_index = np.concatenate([index, depth_map[:, :, None]], axis=-1)
    depth_index = depth_index.reshape(-1, 3)
    xyz = calib.project_image_to_rect(depth_index)
    mask = xyz[:, 2] > 0
    xyz[mask] = roty(t).dot(xyz[mask].T).T
    pad = 479232-xyz.shape[0]
    xyz = np.concatenate([xyz,np.zeros((pad,3))], axis=0)
    depth_index = np.concatenate([depth_index,np.zeros((pad,3))], axis=0)

    return torch.from_numpy(xyz).float(), torch.from_numpy(depth_index[:, :2]).float()


def mask_points(xyz, depth_index, batch_idx):
    """
    filter points
    :param xyz: batch, points, 3
    :param depth_index: batch, points, 2
    :return:
    """
    mask1 = (xyz[batch_idx, :, 0] >= -40) * (xyz[batch_idx, :, 0] < 40) * (xyz[batch_idx, :, 2] > 0) * (
                xyz[batch_idx, :, 2] < 68.8) * (
                    xyz[batch_idx, :, 1] >= -1) * (xyz[batch_idx, :, 1] < 2.5)

    xyz_mask1 = xyz[batch_idx][mask1]
    depth_index_mask1 = depth_index[batch_idx][mask1]
    xz_mask1 = xyz_mask1[:, (0, 2)]
    xz_mask1[:, 0] += 40
    xz_mask1_quant = torch.floor(xz_mask1 * 10)
    return xz_mask1_quant.long(), depth_index_mask1.long()
