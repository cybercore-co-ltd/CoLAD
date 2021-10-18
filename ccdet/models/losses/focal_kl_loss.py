import torch.nn as nn
from mmdet.models import LOSSES


@LOSSES.register_module()
class FocalKLLoss(nn.Module):
    """Focal KL Loss, typically used for knowledge distillation"""

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 gamma=2,
                 loss_weight=1.0):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gamma = gamma

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        raise NotImplementedError
