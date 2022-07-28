import torch
from torch.nn.functional import smooth_l1_loss


def centerness_loss(gt, pred):
    '''
    :param gt:  Bx1xHxW
    :param pred: Bx1xHxW
    :return:
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = - neg_loss
    else:
        loss = - (pos_loss + neg_loss) / num_pos
    return loss


def l1_loss(gt, pred, centermap):
    positives = (centermap == 1).sum(dim=1, keepdim=True)
    loss = positives * smooth_l1_loss(pred, gt,reduction="none")
    return loss.sum()

def non_negative_loss(gt,pred,centermap):
    '''
    This loss function pulls models from predicting cropped boxes. To do that, it penalizes keypoints being
    nearer their centers than its groundtruth counterparts
    :param gt:
    :param pred:
    :param centermap:
    :return:
    '''
    positives = (centermap == 1).sum(dim=1, keepdim=True)
    distance_gt_pred = (pred**2-gt**2)
    zero = torch.zeros_like(distance_gt_pred)
    loss = positives * torch.where(distance_gt_pred>zero,zero,-distance_gt_pred)
    return loss.sum()