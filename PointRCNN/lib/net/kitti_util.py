import torch


class Calib(object):
    def __init__(self, calib):

        self.P2 = torch.from_numpy(calib.P2).cuda()  # 3 x 4
        self.P3 = torch.from_numpy(calib.P3).cuda()  # 3 x 4
        self.R0 = torch.from_numpy(calib.R0).cuda()  # 3 x 3
        self.V2C = torch.from_numpy(calib.V2C).cuda()  # 3 x 4
        self.C2V = torch.from_numpy(calib.C2V).cuda()

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32).cuda()
        pts_hom = torch.cat((pts, ones), dim=1)
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_lidar: (N, 3)
        """
        pts_hom = self.cart_to_hom(torch.matmul(
            pts_rect, torch.inverse(self.R0.t())))
        pts_rect = torch.matmul(pts_hom, self.C2V.t())
        return pts_rect

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = torch.matmul(
            pts_lidar_hom, torch.matmul(self.V2C.t(), self.R0.t()))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = torch.matmul(pts_rect_hom, self.P2.t())
        pts_img = (pts_2d_hom[:, 0:2].t() / pts_rect_hom[:, 2]).t()  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P2.t()[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = torch.cat(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
        return pts_rect

    def img_to_lidar(self, u, v, depth_rect):
        pts_rect = self.img_to_rect(u, v, depth_rect)
        return self.rect_to_lidar(pts_rect)
