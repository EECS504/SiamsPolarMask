import torch.nn.functional as F
import torch
import torch.nn as nn


def my_log_softmax(cls):
    b, a2, h, w = cls.size()
    # N, 2, 25, 25
    cls = cls.view(b, 2, a2 // 2, h, w)
    # N, 2, 1, 25, 25
    cls = cls.permute(0, 2, 3, 4, 1).contiguous()
    # N, 1, 25, 25, 2
    cls = F.log_softmax(cls, dim=4) # The last dim, shrink the scores to range (0,1)
    return cls
class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss

class MaskIOULoss_v2(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v2, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0].clamp(min=1e-6)

        # loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = (l_max / l_min).log().mean(dim=1)
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss

class MaskIOULoss_v3(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v3, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box, 36 lengths
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        # total has shape (N, 36, 2)
        l_max = total.max(dim=2)[0].pow(2) # 0 index is the tensor, 1 index is the arg
        l_min = total.min(dim=2)[0].pow(2)
        # l_max has shape (N, 36)
        # loss = 2 * (l_max.prod(dim=1) / l_min.prod(dim=1)).log()
        # loss = 2 * (l_max.log().sum(dim=1) - l_min.log().sum(dim=1))
        # sum along the vertexes
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss


class My_loss(object):
    """
    This class computes the basic losses.
    """
    def __init__(self):
        # we make use of mask IOU v3 Loss for mask regression,
        # but we found that L1 in log scale can yield a similar performance
        self.mask_reg_loss_func = MaskIOULoss_v3()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def compute_centerness_targets(self, reg_targets):
        # Note that the input must be positive, which is the distance between a vertex and the center.
        minval = torch.min(reg_targets, dim=-1)[0]
        maxval = torch.max(reg_targets, dim=-1)[0]
        centerness = minval / maxval
        return torch.sqrt(centerness)

    def forward(self, cls, mask_reg, centerness, GT_labels, GT_masks):
        """
        Arguments:
            cls (N*2*25*25)
            mask_reg (N*36*25*25)
            centerness (N*1*25*25)
            GT_labels (N*(25*25)*1) binary, 0 or 1, used in classification
            GT_masks (N*(25*25)*36) Distance
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        num_vertexes = 36
        num_cls = 2  # only 2 classes, background and foreground
        mask_reg_flatten = (mask_reg.permute(0, 2, 3, 1).contiguous().view(-1, num_vertexes))
        # mask_reg_flatten (N*25*25)*36
        GT_labels_flatten = (GT_labels.view(-1))
        # GT_labels_flatten (N*25*25)
        cls = my_log_softmax(cls)
        # cls N*25*25*2
        cls_flatten = (cls.view(-1, num_cls))
        # cls_flatten (N*25*25)*2
        GT_masks_flatten = (GT_masks.view(-1, num_vertexes))
        # GT_masks_flatten (N*25*25)*36
        centerness_flatten = (centerness.view(-1))
        # centerness_flatten (N*25*25)

        # I select all positive labels in GT
        pos_inds = torch.nonzero(GT_labels_flatten > 0).squeeze(1)

        mask_reg_flatten = mask_reg_flatten[pos_inds]
        GT_masks_flatten = GT_masks_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        centerness_GT = self.compute_centerness_targets(GT_masks_flatten)
        # Calculate the cls loss for both positive and negative labels
        cls_loss = F.cross_entropy(cls_flatten, GT_labels_flatten)
        # Only calculate IOU loss and centerness loss for positive labels
        if pos_inds.numel() > 0:
            reg_loss = self.mask_reg_loss_func(
                mask_reg_flatten,
                GT_masks_flatten,
                weight=1.0,
                avg_factor=625
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_GT
            )
        else:
            reg_loss = mask_reg_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss