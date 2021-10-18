import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import multiclass_nms
from mmdet.models import HEADS, PAAHead


@HEADS.register_module()
class PAA_COP_Head(PAAHead):
    """PAA Conditional-Objectness-Prediction (COP) head"""

    def __init__(self, coeff_kernel_size=3, *args, **kwargs):
        assert kwargs['loss_cls']['type'] == 'ProbFocalLoss'
        self.coeff_kernel_size = coeff_kernel_size
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        padding = (self.coeff_kernel_size - 1) // 2
        self.coeff_conv = nn.Conv2d(
            self.feat_channels, self.coeff_kernel_size**2, 3, padding=1)
        self.unfold = nn.Unfold(
            [self.coeff_kernel_size, self.coeff_kernel_size],
            padding=padding)

    def init_weights(self):
        super().init_weights()
        normal_init(self.coeff_conv, std=0.01)

    def forward_single(self, x, scale):
        assert self.num_anchors == 1, "only support num anchors = 1"
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat))
        # We predict iou instead of centerness
        iou_pred = self.atss_centerness(reg_feat)

        # cls_score
        N, C, H, W = cls_score.shape
        cls_score = cls_score.view(N*C, 1, H, W)
        cls_score = self.unfold(cls_score)  # ((N*C)x(k*k)x(H*W)
        cls_score = cls_score.permute(0, 2, 1).view(
            N, C, H, W, self.coeff_kernel_size**2)  # NxCxHxWx(k*k)

        # coeffs
        coeffs = self.coeff_conv(reg_feat)  # Nx(k*k)xHxW
        coeffs = coeffs.permute(0, 2, 3, 1).view(
            N, 1, H, W, self.coeff_kernel_size**2)  # Nx1xHxWx(k*k)

        # fusion
        cls_score = torch.sigmoid(cls_score)
        coeffs = torch.sigmoid(coeffs)
        cls_score = (cls_score * coeffs).mean(-1)

        return cls_score, bbox_pred, iou_pred

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        assert with_nms, 'PAA only supports "with_nms=True" now'
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_preds = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_scores, bbox_preds, iou_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * iou_pred[:, None]).sqrt().max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                iou_pred = iou_pred[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_preds.append(iou_pred)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_preds = torch.cat(mlvl_iou_preds)
        mlvl_nms_scores = (mlvl_scores * mlvl_iou_preds[:, None]).sqrt()
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_nms_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=None)
        if self.with_score_voting:
            det_bboxes, det_labels = self.score_voting(det_bboxes, det_labels,
                                                       mlvl_bboxes,
                                                       mlvl_nms_scores,
                                                       cfg.score_thr)

        return det_bboxes, det_labels
