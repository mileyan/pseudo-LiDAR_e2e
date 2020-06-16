import torch
import losswise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metric(object):
    def __init__(self):
        self.RMSELIs = AverageMeter()
        self.RMSELGs = AverageMeter()
        self.ABSRs = AverageMeter()
        self.SQRs = AverageMeter()
        self.DELTA = AverageMeter()
        self.DELTASQ = AverageMeter()
        self.DELTACU = AverageMeter()
        self.losses = AverageMeter()

    def update(self, loss, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu):
        if loss:
            self.losses.update(loss)
        self.RMSELIs.update(RMSE_Linear)
        self.RMSELGs.update(RMSE_Log)
        self.ABSRs.update(abs_relative)
        self.SQRs.update(sq_relative)
        self.DELTA.update(delta)
        self.DELTASQ.update(delta_sq)
        self.DELTACU.update(delta_cu)

    def get_info(self):
        return [self.losses.avg, self.RMSELIs.avg, self.RMSELGs.avg, self.ABSRs.avg, self.SQRs.avg, self.DELTA.avg,
                self.DELTASQ.avg, self.DELTACU.avg]

    def calculate(self, depth, predict, loss=None):
        # only consider 1~80 meters
        mask = (depth >= 1) * (depth <= 80)
        RMSE_Linear = ((((predict[mask] - depth[mask]) ** 2).mean()) ** 0.5).cpu().detach().item()
        RMSE_Log = ((((torch.log(predict[mask]) - torch.log(depth[mask])) ** 2).mean()) ** 0.5).cpu().detach().item()
        abs_relative = (torch.abs(predict[mask] - depth[mask]) / depth[mask]).mean().cpu().detach().item()
        sq_relative = ((predict[mask] - depth[mask]) ** 2 / depth[mask]).mean().cpu().detach().item()
        delta = (torch.max(predict[mask] / depth[mask], depth[mask] / predict[mask]) < 1.25).float().mean().cpu().detach().item()
        delta_sq = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 2).float().mean().cpu().detach().item()
        delta_cu = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 3).float().mean().cpu().detach().item()
        self.update(loss, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu)

    def tensorboard(self, writer, epoch, token='train'):
        writer.add_scalar(token + '/RMSELIs', self.RMSELIs.avg, epoch)
        writer.add_scalar(token + '/RMSELGs', self.RMSELGs.avg, epoch)
        writer.add_scalar(token + '/ABSRs', self.ABSRs.avg, epoch)
        writer.add_scalar(token + '/SQRs', self.SQRs.avg, epoch)
        writer.add_scalar(token + '/DELTA', self.DELTA.avg, epoch)
        writer.add_scalar(token + '/DELTASQ', self.DELTASQ.avg, epoch)
        writer.add_scalar(token + '/DELTACU', self.DELTACU.avg, epoch)

    def print(self, iter, token):
        string = '{}:{}\tL {:.3f} RLI {:.3f} RLO {:.3f} ABS {:.3f} SQ {:.3f} DEL {:.3f} DELQ {:.3f} DELC {:.3f}'.format(token, iter, *self.get_info())
        return string



class LossWise(object):
    def __init__(self, key=None, tag=None, epochs=300):
        self.key = key
        print(self.key)
        if len(self.key)>0:
            losswise.set_api_key(self.key)
            session = losswise.Session(tag=tag, max_iter=epochs)
            self.error = session.graph('Error', kind='min', display_interval=1)
            self.loss = session.graph('Loss', kind='min', display_interval=1)
            self.delta = session.graph('Delta', kind='min', display_interval=1)
            self.session=session

    def update(self, info, epoch, tag='Train'):
        if len(self.key)>0:
            self.loss.append(epoch, {tag+'/loss':info[0]})
            self.error.append(epoch, {tag+'/RMSE':info[1], tag+'/RMSELog':info[2],
                                      tag+'/ABSR':info[3], tag+'/SQUR':info[4]})
            self.delta.append(epoch, {tag+'/1.25':info[5],tag+'/1.25^2':info[6],
                                      tag+'/1.25^3':info[7]})

    def done(self):
        self.session.done()

class LossWise1(object):
    def __init__(self, key=None, tag=None, epochs=300):
        self.key = key
        print(self.key)
        if len(self.key)>0:
            losswise.set_api_key(self.key)
            session = losswise.Session(tag=tag, max_iter=epochs)
            self.error = session.graph('Error', kind='min', display_interval=1)
            self.loss_total = session.graph('Loss', kind='min', display_interval=1)
            self.delta = session.graph('Delta', kind='min', display_interval=1)
            self.session=session

    def update(self, info, epoch, tag='Train'):
        if len(self.key)>0:
            self.loss_total.append(epoch, {tag+'/loss_gt':info[0], tag+'/loss_pseudo':info[1], tag+'/loss_total':info[2]})
            self.error.append(epoch, {tag+'/RMSE':info[3], tag+'/RMSELog':info[4],
                                      tag+'/ABSR':info[5], tag+'/SQUR':info[6]})
            self.delta.append(epoch, {tag+'/1.25':info[7],tag+'/1.25^2':info[8],
                                      tag+'/1.25^3':info[9]})

    def done(self):
        self.session.done()

class Metric1(object):
    def __init__(self):
        self.RMSELIs = AverageMeter()
        self.RMSELGs = AverageMeter()
        self.ABSRs = AverageMeter()
        self.SQRs = AverageMeter()
        self.DELTA = AverageMeter()
        self.DELTASQ = AverageMeter()
        self.DELTACU = AverageMeter()
        self.losses_gt = AverageMeter()
        self.losses_pseudo = AverageMeter()
        self.losses_total = AverageMeter()

    def update(self, loss_gt, loss_pseudo, loss_total, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu):
        self.losses_gt.update(loss_gt)
        self.losses_pseudo.update(loss_pseudo)
        self.losses_total.update(loss_total)
        self.RMSELIs.update(RMSE_Linear)
        self.RMSELGs.update(RMSE_Log)
        self.ABSRs.update(abs_relative)
        self.SQRs.update(sq_relative)
        self.DELTA.update(delta)
        self.DELTASQ.update(delta_sq)
        self.DELTACU.update(delta_cu)

    def get_info(self):
        return [self.losses_gt.avg, self.losses_pseudo.avg, self.losses_total.avg, self.RMSELIs.avg, self.RMSELGs.avg, self.ABSRs.avg, self.SQRs.avg, self.DELTA.avg,
                self.DELTASQ.avg, self.DELTACU.avg]

    def calculate(self, depth, predict, loss_gt=0, loss_psuedo=0, loss_total=0):
        # only consider 1~80 meters
        mask = (depth >= 1) * (depth <= 80)
        RMSE_Linear = ((((predict[mask] - depth[mask]) ** 2).mean()) ** 0.5).cpu().data
        RMSE_Log = ((((torch.log(predict[mask]) - torch.log(depth[mask])) ** 2).mean()) ** 0.5).cpu().data
        abs_relative = (torch.abs(predict[mask] - depth[mask]) / depth[mask]).mean().cpu().data
        sq_relative = ((predict[mask] - depth[mask]) ** 2 / depth[mask]).mean().cpu().data
        delta = (torch.max(predict[mask] / depth[mask], depth[mask] / predict[mask]) < 1.25).float().mean().cpu().data
        delta_sq = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 2).float().mean().cpu().data
        delta_cu = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 3).float().mean().cpu().data
        self.update(loss_gt, loss_psuedo, loss_total, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu)

    def tensorboard(self, writer, epoch, token='train'):
        writer.add_scalar(token + '/RMSELIs', self.RMSELIs.avg, epoch)
        writer.add_scalar(token + '/RMSELGs', self.RMSELGs.avg, epoch)
        writer.add_scalar(token + '/ABSRs', self.ABSRs.avg, epoch)
        writer.add_scalar(token + '/SQRs', self.SQRs.avg, epoch)
        writer.add_scalar(token + '/DELTA', self.DELTA.avg, epoch)
        writer.add_scalar(token + '/DELTASQ', self.DELTASQ.avg, epoch)
        writer.add_scalar(token + '/DELTACU', self.DELTACU.avg, epoch)

    def print(self, iter, token):
        string = '{}:{}\tL {:.3f} {:.3f} {:.3f} RLI {:.3f} RLO {:.3f} ABS {:.3f} SQ {:.3f} DEL {:.3f} DELQ {:.3f} DELC {:.3f}'.format(token, iter, *self.get_info())
        return string



def roty_pth(t):
    ''' Rotation about the y-axis. '''
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.FloatTensor([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

class torchCalib(object):
    def __init__(self, calib, h_shift=0):

        self.P2 = torch.from_numpy(calib.P).cuda().float()  # 3 x 4
        self.P2[1, 2] -= h_shift
#         self.P3 = torch.from_numpy(calib.P3).cuda()  # 3 x 4
        self.R0 = torch.from_numpy(calib.R0).cuda().float()  # 3 x 3
        self.V2C = torch.from_numpy(calib.V2C).cuda().float()  # 3 x 4
        self.C2V = torch.from_numpy(calib.C2V).cuda().float()

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
