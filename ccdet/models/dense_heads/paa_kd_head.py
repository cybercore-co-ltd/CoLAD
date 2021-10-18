from mmdet.models import HEADS, PAAHead, build_loss


@HEADS.register_module()
class PAA_KD_Head(PAAHead):
    """PAA head for Knowledge Distillation"""

    def __init__(self,
                 kd_cls_loss=None,
                 kd_bbox_loss=None,
                 kd_centerness_loss=None,
                 min_iou=0.1,
                 use_pred_bbox=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # buid KD losses
        self.kd_cls_loss = build_loss(kd_cls_loss)
        self.kd_bbox_loss = build_loss(kd_bbox_loss)
        self.kd_centerness_loss = build_loss(kd_centerness_loss)
        self.min_iou = min_iou
        self.use_pred_bbox = use_pred_bbox

    @property
    def with_kd_cls(self):
        """bool: whether the head has kd_cls"""
        return hasattr(self, 'kd_cls_loss') and self.kd_cls_loss is not None

    @property
    def with_kd_bbox(self):
        """bool: whether the head has kd_bbox"""
        return hasattr(self, 'kd_bbox_loss') and \
            self.kd_bbox_loss is not None

    @property
    def with_kd_iou(self):
        """bool: whether the head has kd_bbox"""
        return hasattr(self, 'kd_centerness_loss') and \
            self.kd_centerness_loss is not None
