import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def prob_sigmoid_focal_loss(pred_sigmoid,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred_sigmoid (torch.Tensor): The predicted prob(after sigmoid) with shape (N, C), C is the
            number of classes. 
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_class = pred_sigmoid.shape[-1]
    # Since the mmdet 2.0 has 0-base index ( Coco: 0 for class 1, 80 for background)
    target_onehot = F.one_hot(target, num_class+1)[:, :-1]
    target_onehot = target_onehot.type_as(pred_sigmoid)
    pt = (1 - pred_sigmoid) * target_onehot + \
        pred_sigmoid * (1 - target_onehot)
    focal_weight = (alpha * target_onehot + (1 - alpha) *
                    (1 - target_onehot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred_sigmoid, target_onehot, reduction='none') * focal_weight
    loss = weight_reduce_loss(
        loss, weight.unsqueeze(dim=1), reduction, avg_factor)
    return loss


@LOSSES.register_module()
class ProbFocalLoss(nn.Module):
    """Focal Loss, but receives probabilities as the input instead of logits"""

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(ProbFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The probability prediction (after sigmoid).
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * prob_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
