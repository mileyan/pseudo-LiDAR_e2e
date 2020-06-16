import data.kitti_common as kitti
import data.kitti_util as utils
from data.kitti_object import *
import os
import numpy as np


def predict_kitti_to_file(predictions_dicts, img_idxs, result_save_path, dataset):
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # "predictions_dicts" should be a list of the batch size
    # Each list item is a dictionary with keys "box3d_lidar", "scores". The size of the values equals the number of detected cars

    for i, preds_dict in enumerate(predictions_dicts): # each image

        img_idx = img_idxs[i]
        calib = dataset.get_calibration(img_idx)
        img = dataset.get_image(img_idx)
        image_shape = img.shape[:2]

        if preds_dict["box3d_rect"] is not None:



            scores = preds_dict["scores"]
            box_preds_rect = preds_dict["box3d_rect"]
            result_lines = []

            for box_lidar, score in zip(box_preds_rect, scores): # each box

                theta, cx, cz, w, l = box_lidar

                # pseudo camera space
                R = utils.roty(theta)
                rect_4_points = np.zeros((4, 3))
                rect_4_points[0] = [-l/2.0, 0, -w/2.0]
                rect_4_points[1] = [-l/2.0, 0, w/2.0]
                rect_4_points[2] = [l/2.0, 0, -w/2.0]
                rect_4_points[3] = [l/2.0, 0, w/2.0]
                rect_4_points = np.dot(rect_4_points, np.transpose(R))
                rect_4_points[:, 0] += cx
                rect_4_points[:, 2] += cz

                _, _, z_closest = np.min(rect_4_points, axis=0)

                img_4_points = calib.project_rect_to_image(rect_4_points)
                xmin, _ = np.min(img_4_points, axis=0)
                xmax, _ = np.max(img_4_points, axis=0)
                box_2d = [xmin, 0, xmax, 0]

                if z_closest > 60:
                    box_2d[3] = 10
                elif z_closest > 30:
                    box_2d[3] = 30
                else:
                    box_2d[3] = 50
                # left, top, right, down (<30: 50, 30-60: 30, >60: 10)

                if box_2d[0] > image_shape[1] or box_2d[1] > image_shape[0]:
                    continue
                if box_2d[2] < 0 or box_2d[3] < 0:
                    continue
                box_2d[2:] = np.minimum(box_2d[2:], image_shape[::-1])
                box_2d[:2] = np.maximum(box_2d[:2], [0, 0])

                box_3d = [cx, -1000, cz]

                result_dict = {
                    'name': 'Car',
                    'alpha': -10, # -np.arctan2(-cy, cx) + theta,
                    'bbox': box_2d,
                    'location': box_3d,
                    'dimensions': [0, w, l], # length, height, width or (Harry, to check: h, w, l)
                    'rotation_y': theta,
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []

        result_file = "{}/{}.txt".format(result_save_path, kitti.get_image_index_str(img_idx))
        result_str = '\n'.join(result_lines)

        with open(result_file, 'w') as f:
            f.write(result_str)